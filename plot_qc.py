"""Plot the dolma2_qc score distribution from qc_scan.npz, to pick a filter threshold (light: runs on the login node).

    python plot_qc.py --out plots/dolma2_qc.png

Score = dolma2_qc["1"] (fastText class-1 probability; see experiments.md on which class is "good"); only common_crawl docs have it, other sources are always kept.
Panels (logit x axes): (1) histogram by docs and by tokens, (2) share of CC at or above a threshold, (3) tokens left in
the whole mix for either filter direction, against the 6B training budget, (4) high-score share per WebOrganizer topic. Tokens are estimated from chars
with a per-source tokens/char ratio measured by qc_scan.py.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--scan", default=os.path.expandvars("/work/$USER/dolma3_llama/qc_scan.npz"))
ap.add_argument("--out", default="plots/dolma2_qc.png")
ap.add_argument("--budget", type=float, default=6e9, help="Training tokens to mark (default 6B)")
args = ap.parse_args()

z = np.load(args.scan)
p, chars, dir_id, dir_names = z["p_high"], z["chars"], z["dir"], z["dir_names"]
# topic = directory without its -00NN bucket suffix
names = sorted({d.removeprefix("common_crawl-").rsplit("-", 1)[0] for d in dir_names})
topic = np.array([names.index(d.removeprefix("common_crawl-").rsplit("-", 1)[0]) for d in dir_names], np.int16)[dir_id]
ok = np.isfinite(p)
print(f"{len(p):,} CC docs sampled, {(~ok).sum():,} without dolma2_qc (dropped from the plots)")
p, chars, topic, dir_id = np.clip(p[ok], 0, 1), chars[ok], topic[ok], dir_id[ok]

# Scale the sample to the full mix: per directory, x (total docs / sampled docs, from doc_counts.json); tokens = chars x tokens/char
srcs, st = z["sources"], z["src_stats"].astype(float)
counts = json.load(open(os.path.join(os.path.dirname(args.scan), "doc_counts.json")))
full_docs = {}
for f, n in zip(counts["files"], counts["counts"]):
    full_docs[f.split("/")[1]] = full_docs.get(f.split("/")[1], 0) + n
scale = np.array([full_docs[s] for s in srcs]) / st[:, 0]
tpc = st[:, 2] / st[:, 3]  # tokens per char per source
src_tokens = st[:, 1] * tpc * scale
is_cc = np.array([s.startswith("common_crawl") for s in srcs])
cc_tpc = (st[is_cc, 2].sum() / st[is_cc, 3].sum())
src_scale = dict(zip(srcs, scale))
w = np.array([src_scale[d] for d in dir_names])[dir_id]  # docs each sampled doc stands for
tokens = chars * cc_tpc * w  # full-mix tokens each sampled CC doc stands for
non_cc_tokens = src_tokens[~is_cc].sum()
print(f"full mix (estimated): CC {src_tokens[is_cc].sum() / 1e9:.1f}B tokens, other sources {non_cc_tokens / 1e9:.1f}B")

EPS = 1e-6
p = np.clip(p, EPS, 1 - EPS)  # fastText scores can sit at (or slightly past) 0 / 1; the logit axis needs (0, 1)
thr = 1 / (1 + np.exp(-np.linspace(np.log(EPS / (1 - EPS)), np.log((1 - EPS) / EPS), 2001)))  # logit-spaced thresholds
order = np.argsort(p)
ps = p[order]
cum_t = np.concatenate([[0], np.cumsum(tokens[order])])
cum_d = np.concatenate([[0], np.cumsum(w[order])])
idx = np.searchsorted(ps, thr, side="left")
above_docs = 1 - cum_d[idx] / cum_d[-1]  # share of CC with score >= t
above_tok = 1 - cum_t[idx] / cum_t[-1]
cc_full = cum_t[-1]
left_keep_above = non_cc_tokens + above_tok * cc_full  # keep score >= t ("1" = good)
left_keep_below = non_cc_tokens + (1 - above_tok) * cc_full  # keep score < t ("1" = bad)


def logit_x(a):
    a.set_xscale("logit")
    a.set_xlim(EPS, 1 - EPS)
    a.set_xticks([1e-5, 1e-4, 1e-3, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999])
    a.tick_params(axis="x", labelsize=7, labelrotation=45)


fig, ax = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle(f'dolma2_qc score = ["1"] over common_crawl docs ({len(p):,} sampled; ~{cc_full / 1e9:.0f}B CC tokens of '
             f"~{(cc_full + non_cc_tokens) / 1e9:.0f}B in the mix). x axes are logit-scaled")

a = ax[0, 0]
bins = 1 / (1 + np.exp(-np.linspace(-14, 14, 113)))
a.hist(p, bins=bins, weights=100 * w / w.sum(), alpha=0.6, label="% of docs")
a.hist(p, bins=bins, weights=100 * tokens / tokens.sum(), alpha=0.6, label="% of tokens")
logit_x(a)
a.set_xlabel('dolma2_qc["1"]'), a.set_ylabel("% per bin (0.25 in logit)")
a.set_title("Histogram"), a.legend(), a.grid(alpha=0.3)

a = ax[0, 1]
a.plot(thr, 100 * above_docs, label="docs with score ≥ t")
a.plot(thr, 100 * above_tok, label="tokens with score ≥ t")
for q in (0.5, 0.25, 0.1, 0.05):
    t = thr[np.searchsorted(-above_docs, -q)]
    a.axvline(t, color="gray", ls=":", lw=1)
    a.text(t, 100 * q + 2, f" {q:.0%} docs ≥ {t:.3g}", fontsize=8)
logit_x(a)
a.set_xlabel("threshold t"), a.set_ylabel("% of CC with score ≥ t")
a.set_title("Share of Common Crawl at or above the threshold"), a.legend(), a.grid(alpha=0.3)

a = ax[1, 0]
a.plot(thr, left_keep_above / 1e9, label='keep score ≥ t (if "1" = high quality)')
a.plot(thr, left_keep_below / 1e9, label='keep score < t (if "1" = low quality)')
a.axhline(non_cc_tokens / 1e9, color="gray", ls="--", lw=1, label=f"non-CC only (always kept): {non_cc_tokens / 1e9:.1f}B")
a.axhline(args.budget / 1e9, color="red", ls="--", lw=1, label=f"training budget {args.budget / 1e9:.0f}B")
logit_x(a)
a.set_xlabel("threshold t"), a.set_ylabel("tokens left in the whole mix (B)")
a.set_title("Data left vs threshold"), a.legend(fontsize=8), a.grid(alpha=0.3)

a = ax[1, 1]
med = np.array([np.median(p[topic == i]) for i in range(len(names))])
hi = np.array([np.average(p[topic == i] >= 0.5, weights=tokens[topic == i]) for i in range(len(names))])
share = np.array([tokens[topic == i].sum() for i in range(len(names))]) / tokens.sum()
o = np.argsort(hi)
a.barh(np.arange(len(names)), 100 * hi[o])
a.set_yticks(np.arange(len(names)), [f"{names[i]} ({100 * share[i]:.1f}% of CC tok, median {med[i]:.3f})" for i in o], fontsize=7)
a.set_xlabel("% of the topic's tokens with score ≥ 0.5"), a.set_title("Per WebOrganizer topic"), a.grid(alpha=0.3, axis="x")

fig.tight_layout()
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
fig.savefig(args.out, dpi=110)
print(f"saved {args.out}")

print("\nthreshold t | CC docs ≥ t | CC tokens ≥ t | mix left, keep ≥ t | mix left, keep < t")
for t in (1e-4, 1e-3, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99):
    i = min(np.searchsorted(thr, t), len(thr) - 1)
    print(f"  {t:8.4f}  | {100 * above_docs[i]:6.1f}%     | {100 * above_tok[i]:6.1f}%       | {left_keep_above[i] / 1e9:6.1f}B            | {left_keep_below[i] / 1e9:6.1f}B")
print("\nunweighted quantiles:", {q: round(float(np.quantile(p, q)), 5) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)})
