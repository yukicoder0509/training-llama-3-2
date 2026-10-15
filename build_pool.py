r"""Loss-based soft reweighting, steps 1-3: the scoring pool, its scores, and pool.parquet with a z per row (config.yaml).

    # 1. manifest: CPU, 8 array tasks (files t, t+8, ...)
    sbatch --array=0-7 --gpus-per-node=1 --cpus-per-task=12 --mem=96G --time=2:00:00 --job-name=pool-manifest \
        --output=logs/%x-%A_%a.out --wrap="source ./.venv/bin/activate && export HF_HUB_CACHE=/work/\$USER/hf_cache/hub \
        TOKENIZERS_PARALLELISM=false && python build_pool.py manifest --task \$SLURM_ARRAY_TASK_ID --num_tasks 8"
    # 2. score: 8 GPUs, one process per GPU
    sbatch --gpus-per-node=8 --cpus-per-task=96 --time=2:00:00 --job-name=pool-score --output=logs/%x-%j.out \
        --wrap="source ./.venv/bin/activate && python build_pool.py score"
    # 3. merge: CPU
    sbatch --gpus-per-node=1 --cpus-per-task=12 --mem=200G --time=1:00:00 --job-name=pool-merge --output=logs/%x-%j.out \
        --wrap="source ./.venv/bin/activate && export HF_HUB_CACHE=/work/\$USER/hf_cache/hub && python build_pool.py merge"

manifest: a pool row is any row with hash64(seed, row_id) < target_tokens / (eligible rows x tokens per row) x 2^64
  whose shuffled rank is in [scorer_train_end, dev_start): never a row the scorer trained on, a dev row or an official
  eval row. row_id = the row's global index in file order (not the `id` field), so duplicate rows are drawn
  independently, as in the full mix. Pool rows are tokenized (BOS text EOS) into tokens/<file>.bin; dev rows (ranks
  [dev_start, eval_start)) into dev/<file>.npz. Out: rows/<file>.parquet (row_id, shard, offset, rank, text_hash,
  source, topic, weborganizer_max, original_word_count, n_tokens_full, tok_start). text_hash: metadata.text_hash
  (common_crawl) or a 64-bit blake2b of the text, as prepare_data.py --dedup.
score: one row per unique text_hash (its first row), BOS + first 511 tokens, scorer mean token loss (fp32 cross
  entropy). Out: scores/<file>.parquet (text_hash, mean_loss, n_scored, n_tokens_full) for texts first seen in <file>.
  Shard 0 first checks 20 docs against model(input_ids, labels=input_ids).loss on each doc alone.
merge: scores joined onto every row, rows whose text_hash is a dev text dropped, z = (loss - mean) / std within each
  topic over rows (so duplicates count as in the mix), clipped; n_scored < min_scored -> z = 0. Out: pool.parquet,
  dev.bin, pool_report.txt (share checks, real token count), extremes.txt (top / bottom 10 docs per topic),
  plots/pool_loss_hist.png.
Every stage skips outputs that already exist, so a rerun resumes.
"""

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_data import KEY_MASK, iter_lines, write_tokens  # noqa: E402
from scoring import group_name, parse_dir  # noqa: E402

BOS, EOS = 128000, 128001
cfg = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")))
POOL = os.environ.get("POOL_DIR", cfg["pool"]["dir"])  # POOL_DIR: a test dir

ap = argparse.ArgumentParser()
ap.add_argument("stage", choices=["manifest", "score", "merge"])
ap.add_argument("--task", type=int, default=0)
ap.add_argument("--num_tasks", type=int, default=1)
ap.add_argument("--shard", type=int, default=None, help="score: run only this GPU shard (default: spawn one per GPU)")
ap.add_argument("--num_shards", type=int, default=8)
ap.add_argument("--num_proc", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
ap.add_argument("--batch_tokens", type=int, default=65536)
args = ap.parse_args()
for d in ("rows", "tokens", "dev", "scores"):
    os.makedirs(os.path.join(POOL, d), exist_ok=True)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def hash64(seed, x):
    """splitmix64 of (x + seed * golden), vectorized over uint64 (wrapping arithmetic)."""
    with np.errstate(over="ignore"):
        z = x.astype(np.uint64) + np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


counts = json.load(open(cfg["data"]["doc_counts"]))
assert counts["revision"] == cfg["data"]["revision"]
FILES, PER_FILE = counts["files"], np.array(counts["counts"])
OFFSETS = np.concatenate([[0], np.cumsum(PER_FILE)])
N = int(OFFSETS[-1])
EVAL_START = N - cfg["split"]["num_eval"]
DEV_START = EVAL_START - cfg["split"]["num_dev"]
LO = cfg["split"]["scorer_train_end"]
FRAC = float(cfg["pool"]["target_tokens"]) / ((DEV_START - LO) * float(cfg["pool"]["est_tokens_per_row"]))
THRESH = np.uint64(int(FRAC * 2.0**64))


def row_path(i, kind):
    return os.path.join(POOL, kind, f"{i:04d}" + {"rows": ".parquet", "tokens": ".bin", "dev": ".npz", "scores": ".parquet"}[kind])


# ---------------------------------------------------------------- 1. manifest
tokenizer = None
file_rank = None


def manifest_file(i):
    import pandas as pd
    out = row_path(i, "rows")
    if os.path.exists(out):
        return 0, 0
    rk = file_rank[i]
    rows = np.arange(OFFSETS[i], OFFSETS[i + 1])
    sel = (hash64(cfg["pool"]["seed"], rows) < THRESH) & (rk >= LO) & (rk < DEV_START)
    dev = (rk >= DEV_START) & (rk < EVAL_START)
    src, dir_topic, _ = parse_dir(FILES[i].split("/")[1])
    topic = group_name(src, dir_topic)
    recs, texts, dev_ranks, dev_keys, dev_texts = [], [], [], [], []
    if sel.any() or dev.any():
        for j, line in enumerate(iter_lines(os.path.join(cfg["data"]["root"], FILES[i]))):
            if not (sel[j] or dev[j]):
                continue
            d = json.loads(line)
            m = d.get("metadata") or {}
            th = m.get("text_hash")
            key = int(th) & KEY_MASK if th is not None else int.from_bytes(hashlib.blake2b(d["text"].encode(), digest_size=8).digest(), "little")
            if dev[j]:
                dev_ranks.append(int(rk[j]))
                dev_keys.append(key)
                dev_texts.append(d["text"])
                continue
            wc = m.get("original_word_count")
            recs.append((int(rows[j]), i, j, int(rk[j]), key, src, topic, m.get("weborganizer_max") or "", int(wc) if wc is not None else -1))
            texts.append(d["text"])
        assert j == len(rk) - 1, f"{FILES[i]}: {j + 1} rows, doc_counts says {len(rk)}"

    def tok(ts):
        out_ids = []
        for a in range(0, len(ts), 256):
            for ids in tokenizer(ts[a:a + 256])["input_ids"]:  # starts with BOS
                out_ids.append(np.asarray(ids + [EOS], np.uint32))
        return out_ids

    ids = tok(texts)
    n_tok = np.array([len(x) for x in ids], np.int64)
    with open(row_path(i, "tokens") + ".tmp", "wb") as f:
        for x in ids:
            f.write(x.tobytes())
    os.replace(row_path(i, "tokens") + ".tmp", row_path(i, "tokens"))
    dids = tok(dev_texts)
    np.savez(row_path(i, "dev") + ".tmp.npz", ranks=np.array(dev_ranks, np.int64), keys=np.array(dev_keys, np.uint64), lens=np.array([len(x) for x in dids], np.int64),
             tokens=np.concatenate(dids) if dids else np.zeros(0, np.uint32))
    os.replace(row_path(i, "dev") + ".tmp.npz", row_path(i, "dev"))
    cols = ["row_id", "shard", "offset", "rank", "text_hash", "source", "topic", "weborganizer_max", "original_word_count"]
    df = pd.DataFrame(recs, columns=cols).astype({"row_id": np.int64, "shard": np.int32, "offset": np.int32, "rank": np.int64,
                                                   "text_hash": np.uint64, "original_word_count": np.int32})
    df["n_tokens_full"] = n_tok.astype(np.int32)
    df["tok_start"] = (np.cumsum(n_tok) - n_tok).astype(np.int64)
    df.to_parquet(out + ".tmp", index=False)
    os.replace(out + ".tmp", out)  # the parquet marks the file done
    return len(df), int(n_tok.sum())


def manifest():
    global tokenizer, file_rank
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["data"]["tokenizer"])
    tokenizer.model_max_length = int(1e9)
    assert tokenizer("x")["input_ids"][0] == BOS
    perm = np.random.default_rng(cfg["split"]["shuffle_seed"]).permutation(N)
    rank = np.empty(N, np.int64)
    rank[perm] = np.arange(N)
    del perm
    file_rank = [rank[OFFSETS[i]:OFFSETS[i + 1]] for i in range(len(FILES))]
    todo = [i for i in range(args.task, len(FILES), args.num_tasks) if not os.path.exists(row_path(i, "rows"))]
    log(f"pool fraction {FRAC:.5f} of ranks [{LO:,}, {DEV_START:,}); dev ranks [{DEV_START:,}, {EVAL_START:,}); "
        f"task {args.task}/{args.num_tasks}: {len(todo)} files to do")
    rows = toks = 0
    # biggest files first, so no straggler at the end
    todo.sort(key=lambda i: -PER_FILE[i])
    with Pool(args.num_proc) as p:
        for k, (r, t) in enumerate(p.imap_unordered(manifest_file, todo)):
            rows, toks = rows + r, toks + t
            if (k + 1) % 50 == 0 or k + 1 == len(todo):
                log(f"{k + 1}/{len(todo)} files, {rows:,} pool rows, {toks:,} tokens")


# ---------------------------------------------------------------- 2. score
def load_rows(columns):
    import pandas as pd
    missing = [i for i in range(len(FILES)) if not os.path.exists(row_path(i, "rows"))]
    assert not missing, f"{len(missing)} manifest files missing, e.g. {missing[:5]}"
    return pd.concat([pd.read_parquet(row_path(i, "rows"), columns=columns) for i in range(len(FILES))], ignore_index=True)


def score_shard(shard):
    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM
    from scoring import score_stream
    P = cfg["scorer"]["prefix_tokens"]
    uniq = pd.read_parquet(os.path.join(POOL, "unique.parquet"))
    files = [i for i in np.unique(uniq.shard) if i % args.num_shards == shard and not os.path.exists(row_path(i, "scores"))]
    torch.cuda.set_device(shard % torch.cuda.device_count())
    model = AutoModelForCausalLM.from_pretrained(cfg["scorer"]["model"], attn_implementation="flash_attention_3", dtype=torch.bfloat16).cuda().eval()
    log(f"shard {shard}: {len(files)} files to score")
    groups = dict(tuple(uniq.groupby("shard")))
    checked = shard != 0
    for k, i in enumerate(files):
        u = groups[i]
        t = np.memmap(row_path(i, "tokens"), dtype=np.uint32, mode="r")
        n = np.minimum(u.n_tokens_full.to_numpy(), P)
        start = u.tok_start.to_numpy()
        buf = np.concatenate([np.asarray(t[s:s + m]) for s, m in zip(start, n)])
        assert (buf[np.concatenate([[0], np.cumsum(n)[:-1]])] == BOS).all()
        doc_start = np.concatenate([[0], np.cumsum(n)])
        loss_sum, pred = score_stream(model, buf, doc_start, window=P, batch_tokens=args.batch_tokens, log=lambda *a: None)
        mean = np.where(pred > 0, loss_sum / np.maximum(pred, 1), np.nan)
        if not checked:  # batched varlen loss == plain forward on each doc alone
            big = np.flatnonzero(n >= 2)[:20]
            with torch.no_grad():
                ref = [model(input_ids=torch.from_numpy(buf[doc_start[d]:doc_start[d + 1]].astype(np.int64))[None].cuda(),
                             labels=torch.from_numpy(buf[doc_start[d]:doc_start[d + 1]].astype(np.int64))[None].cuda()).loss.item() for d in big]
            diff = np.abs(np.array(ref) - mean[big])
            log(f"check: 20 docs, max |batched - single| = {diff.max():.2e}")
            assert diff.max() < 1e-3, f"batched loss differs from single-doc loss: {diff}"
            checked = True
        pd.DataFrame({"text_hash": u.text_hash.to_numpy(), "mean_loss": mean.astype(np.float32), "n_scored": pred.astype(np.int32),
                      "n_tokens_full": u.n_tokens_full.to_numpy()}).to_parquet(row_path(i, "scores") + ".tmp", index=False)
        os.replace(row_path(i, "scores") + ".tmp", row_path(i, "scores"))
        if (k + 1) % 100 == 0:
            log(f"shard {shard}: {k + 1}/{len(files)} files")
    log(f"shard {shard}: done")


def score():
    if args.shard is not None:
        return score_shard(args.shard)
    uq = os.path.join(POOL, "unique.parquet")
    if not os.path.exists(uq):
        rows = load_rows(["shard", "text_hash", "n_tokens_full", "tok_start"])
        uniq = rows.drop_duplicates("text_hash")  # first row of each text in file order
        log(f"{len(rows):,} pool rows, {len(uniq):,} unique texts ({1 - len(uniq) / len(rows):.2%} of rows are repeats)")
        uniq.to_parquet(uq + ".tmp", index=False)
        os.replace(uq + ".tmp", uq)
    procs = [subprocess.Popen([sys.executable, __file__, "score", "--shard", str(s), "--num_shards", str(args.num_shards),
                               "--batch_tokens", str(args.batch_tokens)]) for s in range(args.num_shards)]
    codes = [p.wait() for p in procs]
    assert not any(codes), f"score shards failed: {codes}"


# ---------------------------------------------------------------- 3. merge
def merge():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from transformers import AutoTokenizer
    rows = load_rows(None)
    # a file has a score file iff it holds the first row of some text (unique.parquet); others only repeat texts
    scored = np.unique(pd.read_parquet(os.path.join(POOL, "unique.parquet"), columns=["shard"]).shard)
    missing = [i for i in scored if not os.path.exists(row_path(i, "scores"))]
    assert not missing, f"{len(missing)} score files missing"
    scores = pd.concat([pd.read_parquet(row_path(i, "scores"), columns=["text_hash", "mean_loss", "n_scored"]) for i in scored], ignore_index=True)
    assert scores.text_hash.is_unique
    pool = rows.merge(scores, on="text_hash", how="left", validate="many_to_one")
    assert pool.mean_loss.notna().sum() == (pool.n_scored > 0).sum() and pool.n_scored.notna().all()
    rep = []

    def say(s=""):
        print(s, flush=True)
        rep.append(s)

    # dev set: 10k rows right before the official eval, in rank order
    dev_parts = [np.load(row_path(i, "dev")) for i in range(len(FILES))]
    dranks = np.concatenate([z["ranks"] for z in dev_parts])
    dlens = np.concatenate([z["lens"] for z in dev_parts])
    dtoks = np.split(np.concatenate([z["tokens"] for z in dev_parts]), np.cumsum(dlens)[:-1])
    assert len(dranks) == cfg["split"]["num_dev"]
    order = np.argsort(dranks)
    if not os.path.exists(os.path.join(POOL, "dev.bin")):
        write_tokens(os.path.join(POOL, "dev.bin"), [dtoks[k] for k in order])
    dev_keys = np.concatenate([z["keys"] for z in dev_parts])
    say(f"dev: {len(dranks):,} rows, {dlens.sum():,} tokens -> {POOL}/dev.bin")
    in_dev = np.isin(pool.text_hash.to_numpy(), dev_keys)  # pool rows whose text is a dev text
    say(f"pool rows whose text_hash is a dev text: {in_dev.sum():,} (dropped)")
    pool = pool[~in_dev].reset_index(drop=True)

    # z within topic, from row-level stats of the trusted rows
    trusted = pool.n_scored >= cfg["scorer"]["min_scored"]
    st = pool[trusted].groupby("topic").mean_loss.agg(["mean", "std", "size"])
    mu, sd = pool.topic.map(st["mean"]), pool.topic.map(st["std"])
    c = cfg["pool"]["z_clip"]
    pool["z"] = np.where(trusted, ((pool.mean_loss - mu) / sd).clip(-c, c), 0.0).astype(np.float32)
    pool["z"] = pool.z.fillna(0.0)
    cols = ["row_id", "shard", "offset", "text_hash", "topic", "source", "weborganizer_max", "n_tokens_full", "tok_start",
            "mean_loss", "n_scored", "z"]
    pool[cols].to_parquet(os.path.join(POOL, "pool.parquet.tmp"), index=False)
    os.replace(os.path.join(POOL, "pool.parquet.tmp"), os.path.join(POOL, "pool.parquet"))

    T = pool.n_tokens_full.sum()
    say(f"pool: {len(pool):,} rows, {pool.text_hash.nunique():,} unique texts, {T:,} tokens "
        f"(target {cfg['pool']['target_tokens']:.3g}; {T / len(pool):.1f} tokens/row vs est {cfg['pool']['est_tokens_per_row']})")
    if not 18e9 <= T <= 20e9:
        say(f"WARNING: pool outside 18-20B tokens; set est_tokens_per_row = {T / len(pool):.1f} and rebuild")
    say(f"untrusted (n_scored < {cfg['scorer']['min_scored']}): {(~trusted).mean():.2%} of rows, "
        f"{pool.n_tokens_full[~trusted.to_numpy()].sum() / T:.2%} of tokens -> z = 0")
    say(f"z clipped at +-{c}: {(pool.z.abs() >= c).mean():.3%} of rows")

    # Check: topic shares of rows match the full mix (all rows, from doc_counts) within 0.1 pp
    full = pd.Series(PER_FILE, index=[group_name(*parse_dir(f.split("/")[1])[:2]) for f in FILES]).groupby(level=0).sum()
    full = full / full.sum()
    mine = pool.topic.value_counts(normalize=True).reindex(full.index).fillna(0)
    dpp = (mine - full) * 100
    say(f"\ntopic row shares vs full mix: max |diff| {dpp.abs().max():.3f} pp over {len(full)} topics "
        f"({'OK' if dpp.abs().max() <= 0.1 else 'FAIL'}: <= 0.1 pp)")
    tok_share = pool.groupby("topic").n_tokens_full.sum() / T
    say(f"{'topic':<34} {'rows % mix':>10} {'rows % pool':>11} {'diff pp':>8} {'tokens %':>9} {'mean loss':>9} {'std':>6}")
    for g in full.sort_values(ascending=False).index:
        say(f"{g:<34} {full[g] * 100:>10.3f} {mine[g] * 100:>11.3f} {dpp[g]:>8.3f} {tok_share.get(g, 0) * 100:>9.3f} "
            f"{st['mean'].get(g, np.nan):>9.3f} {st['std'].get(g, np.nan):>6.3f}")
    open(os.path.join(POOL, "pool_report.txt"), "w").write("\n".join(rep) + "\n")

    # Loss histograms: overall and per topic
    topics = list(full.sort_values(ascending=False).index)
    ncol = 8
    fig, axes = plt.subplots(-(-(len(topics) + 1) // ncol), ncol, figsize=(3 * ncol, 2.2 * (-(-(len(topics) + 1) // ncol))))
    axes = axes.ravel()
    x = pool.mean_loss[trusted.to_numpy()]
    bins = np.linspace(0, 7, 71)
    axes[0].hist(x, bins=bins, color="k")
    axes[0].set_title("ALL", fontsize=8)
    for ax, g in zip(axes[1:], topics):
        ax.hist(pool.mean_loss[trusted.to_numpy() & (pool.topic == g).to_numpy()], bins=bins)
        ax.set_title(g, fontsize=8)
    for ax in axes:
        ax.tick_params(labelsize=6)
    fig.suptitle("Scorer mean loss (first 512 tokens) per row, trusted rows")
    fig.tight_layout()
    os.makedirs("plots", exist_ok=True)
    fig.savefig("plots/pool_loss_hist.png", dpi=110)

    # Top / bottom 10 unique texts per topic, for a read
    tok = AutoTokenizer.from_pretrained(cfg["data"]["tokenizer"])
    u = pool[trusted.to_numpy()].drop_duplicates("text_hash")
    with open(os.path.join(POOL, "extremes.txt"), "w") as f:
        for g in topics:
            ug = u[u.topic == g].sort_values("mean_loss")
            for name, part in (("LOWEST", ug.head(10)), ("HIGHEST", ug.tail(10))):
                f.write(f"\n===== {g} {name} loss\n")
                for r in part.itertuples():
                    t = np.memmap(row_path(r.shard, "tokens"), dtype=np.uint32, mode="r")
                    text = tok.decode(np.asarray(t[r.tok_start + 1:r.tok_start + min(r.n_tokens_full, 120)]).tolist())
                    f.write(f"[{r.mean_loss:.3f} z={r.z:+.2f} {r.n_tokens_full} tok] {text[:300]!r}\n")
    log(f"wrote pool.parquet, pool_report.txt, extremes.txt in {POOL}; plots/pool_loss_hist.png")


{"manifest": manifest, "score": score, "merge": merge}[args.stage]()
