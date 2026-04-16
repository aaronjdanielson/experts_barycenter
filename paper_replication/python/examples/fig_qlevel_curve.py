"""
Fig 3 — Learned quantile curve: q_θ(D) = σ(0.65 − 2.09 D).
Shows q=0.5 (BC) reference, crossover, and rug of OOS dispersion values.
Output: figures/fig_qlevel_curve.pdf
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import expit

# ── parameters ───────────────────────────────────────────────────────────────
alpha_hat, beta_hat = 0.65, -2.09
D_cross = -alpha_hat / beta_hat          # ≈ 0.311

# ── OOS dispersion values ─────────────────────────────────────────────────────
OOS_CSV = Path(__file__).parent.parent.parent / "output" / "thread3" / "oos_results.csv"
oos = pd.read_csv(OOS_CSV)
d_oos = oos["disp"].values

# ── curve ────────────────────────────────────────────────────────────────────
d_range = np.linspace(0.0, 0.75, 400)
q_vals  = expit(alpha_hat + beta_hat * d_range)

# ── figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 4.0))

# shaded regions
ax.axvspan(0, D_cross,        alpha=0.06, color="#2196F3", label="$q > 0.5$: above median")
ax.axvspan(D_cross, d_range[-1], alpha=0.06, color="#E91E63", label="$q < 0.5$: below median")

# BC reference
ax.axhline(0.5, color="black", lw=1.0, ls="--", label="$q = 0.5$ (fixed BC)")

# crossover
ax.axvline(D_cross, color="#888", lw=0.85, ls=":")
ax.annotate(f"$D^* \\approx {D_cross:.2f}$",
            xy=(D_cross + 0.01, 0.53), fontsize=8.5, color="#555")

# learned curve
ax.plot(d_range, q_vals, color="#E91E63", lw=2.5,
        label=r"$q_\theta(D) = \sigma(0.65 - 2.09\,D)$")

# annotation points
for D_annot, label, va in [(0.05, "$q\\approx0.66$\n(low $D$)", "bottom"),
                            (0.55, "$q\\approx0.40$\n(high $D$)", "top")]:
    q_a = float(expit(alpha_hat + beta_hat * D_annot))
    ax.plot(D_annot, q_a, 'o', color="#E91E63", ms=6, zorder=5)
    ax.annotate(label, xy=(D_annot, q_a),
                xytext=(D_annot + 0.03, q_a + (0.04 if va=="bottom" else -0.04)),
                fontsize=8, color="#E91E63", va=va)

# rug
ax.plot(d_oos, [0.02] * len(d_oos), "|", color="#888", alpha=0.6, ms=10, lw=1.2,
        label="OOS quarters")

ax.set_xlabel(r"Cross-sectional dispersion $D_t$", fontsize=10)
ax.set_ylabel(r"Learned quantile level $q_\theta(D_t)$", fontsize=10)
ax.set_xlim(-0.01, 0.76)
ax.set_ylim(0.02, 0.98)
ax.tick_params(labelsize=9)
ax.legend(fontsize=8.5, loc="upper right", framealpha=0.9)

fig.tight_layout()

# ── save ─────────────────────────────────────────────────────────────────────
OUT_FIG = Path(__file__).parent.parent.parent.parent / "figures" / "fig_qlevel_curve.pdf"
fig.savefig(OUT_FIG, bbox_inches="tight")
print(f"Saved {OUT_FIG}")
