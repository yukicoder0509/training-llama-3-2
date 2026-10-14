"""MiniCPM-style data scaling: fit final eval loss vs training tokens D, one WSD run per budget (light: login node).

    python fit_scaling.py --out plots/data_scaling.png

Forms: power law L(D) = L_inf + A * D^-alpha, and exponential L(D) = L_inf + A * exp(-D / tau) (D in B tokens).
Hold-out check: fit Sophia on D <= 2B and predict its 3B run. Final eval losses come from W&B (experiments.md 2026-10-14).
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (tokens in B, final eval loss); 1.1B = two seeds. Same settings per optimizer, original (non-dedup) data, 0.5M batch
RUNS = {
    "Sophia (lr 5e-5, rho 0.05, wd 0.2)": [(0.5, 4.021), (1.0, 3.387), (1.1, 3.320), (1.1, 3.323), (1.5, 3.176), (2.0, 3.047), (3.0, 2.897)],
    "AdamW (lr 3e-4, wd 0.1)": [(0.5, 4.256), (1.0, 3.730), (1.1, 3.637), (1.5, 3.497), (2.0, 3.354)],
}
FORMS = {
    "power": (lambda d, l, a, k: l + a * d ** -k, np.linspace(0.05, 5, 4000)),  # k = alpha
    "exp": (lambda d, l, a, k: l + a * np.exp(-d / k), np.linspace(0.05, 10, 4000)),  # k = tau (B tokens)
}


def fit(form, pts):
    # least squares: for each k on the grid, L_inf and A are linear (basis f(d, 0, 1, k)); keep the best k with L_inf >= 0
    f, ks = FORMS[form]
    d, y = np.array(pts).T
    best = None
    for k in ks:
        X = np.stack([np.ones_like(d), f(d, 0, 1, k)], 1)
        (l, a), *_ = np.linalg.lstsq(X, y, rcond=None)
        sse = np.sum((X @ [l, a] - y) ** 2)
        if l >= 0 and a > 0 and (best is None or sse < best[0]):
            best = (sse, (l, a, k))
    p = best[1]
    rmse = np.sqrt(np.mean((f(d, *p) - y) ** 2))
    return p, rmse


ap = argparse.ArgumentParser()
ap.add_argument("--out", default="plots/data_scaling.png")
args = ap.parse_args()

fig, ax = plt.subplots(figsize=(9, 6))
D = np.linspace(0.4, 6.5, 300)
for (name, pts), c in zip(RUNS.items(), ("C0", "C1")):
    d, y = np.array(pts).T
    ax.scatter(d, y, color=c, zorder=3, label=name)
    for form, ls in (("power", "-"), ("exp", "--")):
        p, rmse = fit(form, pts)
        f = FORMS[form][0]
        print(f"{name:36s} {form:5s} L_inf={p[0]:.3f} A={p[1]:.3f} k={p[2]:.3f}  rmse={rmse:.4f}  "
              + "  ".join(f"{x}B:{f(x, *p):.3f}" for x in (3, 4, 6)))
        ax.plot(D, f(D, *p), c, ls=ls, lw=1.2, label=f"{form}: L_inf {p[0]:.2f}, 6B {f(6, *p):.3f}")
    if max(d) > 2:  # hold-out: fit on <= 2B, predict the largest run
        held = [q for q in pts if q[0] <= 2]
        for form in FORMS:
            p, _ = fit(form, held)
            print(f"  hold-out {form:5s} fit on <=2B: predicts {max(d)}B = {FORMS[form][0](max(d), *p):.3f} (actual {y[d.argmax()]:.3f}), "
                  f"6B = {FORMS[form][0](6, *p):.3f}")
ax.axhline(np.log(20), color="gray", ls=":", lw=1, label="ppl 20")
ax.set_xlabel("training tokens (B)"), ax.set_ylabel("final eval loss")
ax.set_title("Data scaling, one WSD run per budget (solid: power law, dashed: exponential)")
ax.set_ylim(2.4, 4.4), ax.grid(alpha=0.3), ax.legend(fontsize=8)
fig.tight_layout()
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
fig.savefig(args.out, dpi=110)
print(f"saved {args.out}")
