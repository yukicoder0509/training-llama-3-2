"""Continuous proxy metrics from lm-eval per-sample logs (DataDecide-style), vs 0/1 accuracy.

Reads OUT_DIR/<model>/<suite>/**/samples_<task>_*.jsonl written by eval_harness.sbatch (--log_samples): each multiple
choice question has the log-likelihood ll_i of every choice's text given the prompt. Per question:
  acc / acc_norm      argmax ll_i (/ chars_i) == gold (lm-eval's own metrics, recomputed)
  logp_char           ll_gold / chars_gold: log-likelihood per character of the correct answer
  prob_norm           exp(ll_gold) / sum_i exp(ll_i): probability of the correct answer among the choices
  margin_char         ll_gold / chars_gold - max_{i != gold} ll_i / chars_i
MMLU subtasks are pooled per suite. LAMBADA (one continuation) gives acc and logp_char only.
With --ref, each model is also compared to the reference model question by question: delta, paired stderr and
z = delta / stderr, so a metric's resolving power is visible as |z| (accuracy vs the continuous metrics).

Usage: python harness_proxy.py /work/$USER/harness_samples --ref proxy-dyn-normuon-1b-bs128 [--models a b ...]
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

METRICS = ["acc", "acc_norm", "logp_char", "prob_norm", "margin_char"]


def gold_index(rec, choices):
    t = rec["target"]
    if isinstance(t, int) or (isinstance(t, str) and t.strip().lstrip("-").isdigit()):
        return int(t)
    return choices.index(t if t in choices else " " + str(t))


def question_metrics(rec):
    args = [rec["arguments"][k] for k in sorted(rec["arguments"], key=lambda k: int(k.rsplit("_", 1)[1]))]
    conts = [a["arg_1"] for a in args]
    ll = np.array([float(r[0]) for r in rec["filtered_resps"]])
    if len(ll) == 1:  # loglikelihood task (LAMBADA): the target word given the passage
        return {"acc": float(rec["filtered_resps"][0][1] == "True"), "logp_char": ll[0] / len(conts[0])}
    g = gold_index(rec, conts)
    chars = np.array([max(1, len(c)) for c in conts], dtype=float)
    per_char = ll / chars
    p = np.exp(ll - ll.max())
    wrong = np.delete(per_char, g)
    return {"acc": float(ll.argmax() == g), "acc_norm": float(per_char.argmax() == g), "logp_char": per_char[g],
            "prob_norm": p[g] / p.sum(), "margin_char": per_char[g] - wrong.max()}


def load_model(model_dir):
    """{task: {doc_key: {metric: value}}}; MMLU subtasks pooled under their suite name."""
    out = defaultdict(dict)
    for f in glob.glob(f"{model_dir}/*/**/samples_*.jsonl", recursive=True):
        suite = os.path.relpath(f, model_dir).split(os.sep)[0]
        task = re.match(r"samples_(.+)_\d{4}-\d\d-\d\dT", os.path.basename(f)).group(1)
        key = suite if task.startswith("mmlu") else task
        for line in open(f):
            rec = json.loads(line)
            out[key][(task, rec["doc_id"])] = question_metrics(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--ref", action="append", default=[], help="reference model(s); others are paired against the "
                    "first ref whose name prefix (proxy- or not) matches")
    ap.add_argument("--out", default=None, help="markdown output (default: print)")
    a = ap.parse_args()
    models = a.models or sorted(d for d in os.listdir(a.root) if os.path.isdir(f"{a.root}/{d}"))
    data = {m: load_model(f"{a.root}/{m}") for m in models}
    tasks = sorted({t for d in data.values() for t in d})
    lines = []
    for t in tasks:
        lines += [f"\n### {t}\n", "| Model | n | " + " | ".join(METRICS) + " |", "|---" * (len(METRICS) + 2) + "|"]
        for m in models:
            q = data[m].get(t, {})
            if not q:
                continue
            row = []
            for k in METRICS:
                v = [x[k] for x in q.values() if k in x]
                row.append(f"{np.mean(v):.4f}" if v else "")
            lines.append(f"| {m} | {len(q)} | " + " | ".join(row) + " |")
        for ref in a.ref:
            if ref not in data or t not in data[ref]:
                continue
            lines += [f"\nPaired vs `{ref}`: delta (z = delta / paired stderr)\n",
                      "| Model | " + " | ".join(METRICS) + " |", "|---" * (len(METRICS) + 1) + "|"]
            for m in models:
                if m == ref or t not in data[m] or m.startswith("proxy") != ref.startswith("proxy"):
                    continue
                keys = sorted(set(data[m][t]) & set(data[ref][t]))
                row = []
                for k in METRICS:
                    d = np.array([data[m][t][x][k] - data[ref][t][x][k] for x in keys if k in data[m][t][x]])
                    if len(d) < 2:
                        row.append("")
                        continue
                    se = d.std(ddof=1) / np.sqrt(len(d))
                    row.append(f"{d.mean():+.4f} ({d.mean() / se:+.1f})" if se > 0 else f"{d.mean():+.4f}")
                lines.append(f"| {m} | " + " | ".join(row) + " |")
    text = "\n".join(lines)
    if a.out:
        open(a.out, "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
