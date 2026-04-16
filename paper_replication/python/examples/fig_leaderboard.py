"""
Fig 4 — OOS performance leaderboard.
Two evaluation windows side by side: 56Q (blend) and 32Q (Q-Level).
Output: figures/fig_leaderboard.pdf
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── known values (from paper tables) ─────────────────────────────────────────
# 56-quarter OOS window (2008–2025), Wilcoxon p vs AM from Tab tab:blend
w56 = {
    "AM":    0.423,
    "BC":    0.411,
    "TM":    0.409,
    "Blend": 0.404,
    "Q-Level": np.nan,
}

# 32-quarter OOS window (2017–2024), from oos_results.csv
OOS_CSV = Path(__file__).parent.parent.parent / "output" / "thread3" / "oos_results.csv"
oos = pd.read_csv(OOS_CSV)
w32 = {
    "AM":      oos["AM"].mean(),
    "BC":      oos["BC"].mean(),
    "TM":      oos["TM"].mean(),
    "Q-Level": oos["QLevel"].mean(),
    "Blend":   np.nan,
}

methods = ["AM", "BC", "TM", "Blend", "Q-Level"]
labels  = ["AM", "BC", "TM", "Adaptive\nBlend", "Q-Level"]
colors  = ["#9E9E9E", "#2196F3", "#FF9800", "#4CAF50", "#E91E63"]

v56 = [w56[m] for m in methods]
v32 = [w32[m] for m in methods]

# ── figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.5, 4.8))

x     = np.arange(len(methods))
width = 0.38

bars56 = ax.bar(x - width/2, v56, width, color=colors, alpha=0.92,
                label="56Q window (2008–2025)", zorder=3)
bars32 = ax.bar(x + width/2, v32, width, color=colors, alpha=0.45,
                edgecolor=colors, linewidth=1.5, label="32Q window (2017–2024)", zorder=3)

# % vs AM annotation inside bars
am56 = w56["AM"]
am32 = w32["AM"]
for i, m in enumerate(methods):
    # 56Q bar
    if not np.isnan(v56[i]):
        pct = (am56 - v56[i]) / am56 * 100
        txt = f"{pct:+.1f}%" if m != "AM" else "ref"
        ypos = v56[i] - 0.004
        ax.text(x[i] - width/2, ypos, txt, ha="center", va="top",
                fontsize=7.5, color="white" if m != "AM" else "#555",
                fontweight="bold")
    # 32Q bar
    if not np.isnan(v32[i]):
        pct = (am32 - v32[i]) / am32 * 100
        txt = f"{pct:+.1f}%" if m != "AM" else "ref"
        ypos = v32[i] - 0.004
        ax.text(x[i] + width/2, ypos, txt, ha="center", va="top",
                fontsize=7.5, color="#444" if m != "AM" else "#888",
                fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("Mean OOS RPS (lower is better)", fontsize=10)
ax.set_ylim(0.35, 0.55)
ax.tick_params(labelsize=9)
ax.yaxis.grid(True, lw=0.5, alpha=0.4, zorder=0)
ax.set_axisbelow(True)

ax.legend(fontsize=9, framealpha=0.9)

# Q-Level callout
ax.annotate("$-15.5\\%$ vs BC\n$p < 0.001$",
            xy=(x[-1] + width/2, w32["Q-Level"]),
            xytext=(x[-1] + width/2 + 0.35, w32["Q-Level"] + 0.025),
            fontsize=8.5, color="#E91E63",
            arrowprops=dict(arrowstyle="->", color="#E91E63", lw=1.2))

fig.tight_layout()

# ── save ─────────────────────────────────────────────────────────────────────
OUT_FIG = Path(__file__).parent.parent.parent.parent / "figures" / "fig_leaderboard.pdf"
fig.savefig(OUT_FIG, bbox_inches="tight")
print(f"Saved {OUT_FIG}")
print(f"  56Q: AM={am56:.3f}  BC={w56['BC']:.3f}  TM={w56['TM']:.3f}  "
      f"Blend={w56['Blend']:.3f}")
print(f"  32Q: AM={am32:.3f}  BC={w32['BC']:.3f}  TM={w32['TM']:.3f}  "
      f"Q-Level={w32['Q-Level']:.3f}")
