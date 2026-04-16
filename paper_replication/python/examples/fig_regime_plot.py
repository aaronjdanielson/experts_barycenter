"""
Fig 2 — Regime plot: per-quarter ΔRPS (BC − AM) vs. cross-sectional dispersion D_t.
Overlays Prop C2' theoretical prediction curve.
Output: figures/fig_regime_plot.pdf
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

# ── load data ────────────────────────────────────────────────────────────────
OUT_CSV = Path(__file__).parent / "output" / "spf_backtest" / "blend_oos.csv"
oos = pd.read_csv(OUT_CSV)

delta = oos["rps_bc"] - oos["rps_am"]   # positive = AM wins, negative = BC wins

# ── Prop C2' theory curve ─────────────────────────────────────────────────────
# ΔRPS ≈ ε²M² − (π/2 − 1)·σ²/n
# Calibrate: crossover observed at D ≈ 0.30 in data
# Use ε=0.15, M=1.0, n=40 (training panel size)
eps, M, n = 0.15, 1.0, 40
d_range = np.linspace(0.0, oos["disp"].max() * 1.15, 300)
theory = eps**2 * M**2 - (np.pi / 2 - 1) * d_range / n   # D_t = σ² in our notation

# ── figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))

colors = ["#E91E63" if d < 0 else "#9E9E9E" for d in delta]
ax.scatter(oos["disp"], delta, c=colors, s=28, alpha=0.85, zorder=3,
           linewidths=0)

ax.axhline(0, color="black", lw=0.9, ls="--", zorder=2)

ax.plot(d_range, theory, color="#F9A825", lw=2.2, zorder=4,
        label=r"Prop.\ C2$'$: $\varepsilon^2M^2 - \tfrac{\pi/2-1}{n}\sigma^2$")

# crossover marker
D_cross = eps**2 * M**2 * n / (np.pi / 2 - 1)
ax.axvline(D_cross, color="#F9A825", lw=0.8, ls=":", zorder=2)

# legend proxies for scatter
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#E91E63',
           markersize=7, label='BC wins'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#9E9E9E',
           markersize=7, label='AM wins'),
    Line2D([0], [0], color='#F9A825', lw=2.2,
           label=r"Prop.\ C2$'$ prediction"),
]
ax.legend(handles=legend_elements, fontsize=8.5, framealpha=0.9)

ax.set_xlabel(r"Cross-sectional dispersion $D_t$", fontsize=10)
ax.set_ylabel(r"$\Delta\mathrm{RPS}$ (BC $-$ AM)", fontsize=10)
ax.tick_params(labelsize=9)

n_bc = (delta < 0).sum()
n_am = (delta >= 0).sum()
ax.annotate(f"BC wins: {n_bc}/{len(delta)} quarters",
            xy=(0.03, 0.05), xycoords="axes fraction", fontsize=8.5,
            color="#E91E63")

fig.tight_layout()

# ── save ─────────────────────────────────────────────────────────────────────
OUT_FIG = Path(__file__).parent.parent.parent.parent / "figures" / "fig_regime_plot.pdf"
fig.savefig(OUT_FIG, bbox_inches="tight")
print(f"Saved {OUT_FIG}")
