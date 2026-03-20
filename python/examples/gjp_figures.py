"""
Generates fig_gjp1_brier.pdf from output/gjp/gjp_backtest.csv.

CSV columns: n, realized, p_am, p_bc, dispersion, bs_am, bs_bc
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

GJP_DIR = Path("output/gjp")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

df = pd.read_csv(GJP_DIR / "gjp_backtest.csv")
print(f"GJP questions: {len(df)}")

# Dispersion terciles
terciles = pd.qcut(df["dispersion"], 3, labels=["Low", "Mid", "High"])
df["tercile"] = terciles
TERC_COLORS = {"Low": "#1f77b4", "Mid": "#ff7f0e", "High": "#d62728"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# — Left: scatter AM vs BC Brier score —
for terc, grp in df.groupby("tercile"):
    ax1.scatter(grp["bs_am"], grp["bs_bc"],
                color=TERC_COLORS[terc], s=18, alpha=0.7, label=f"{terc} dispersion")
lo = 0
hi = max(df["bs_am"].max(), df["bs_bc"].max()) * 1.05
ax1.plot([lo, hi], [lo, hi], "k--", lw=0.9, label="BC = AM")
ax1.set_xlabel("AM Brier score")
ax1.set_ylabel("BC Brier score")
ax1.set_title("AM vs. BC Brier score per question")
ax1.legend(framealpha=0.9, loc="upper left")
ax1.set_xlim(lo, hi)
ax1.set_ylim(lo, hi)

# — Right: mean Brier by tercile —
terc_order = ["Low", "Mid", "High"]
am_means = [df[df["tercile"] == t]["bs_am"].mean() for t in terc_order]
bc_means = [df[df["tercile"] == t]["bs_bc"].mean() for t in terc_order]
x = np.arange(len(terc_order))
w = 0.35
ax2.bar(x - w/2, am_means, width=w, color="#d62728", alpha=0.85, label="AM")
ax2.bar(x + w/2, bc_means, width=w, color="#1f77b4", alpha=0.85, label="BC")
ax2.set_xticks(x)
ax2.set_xticklabels([f"{t}\ndispersion" for t in terc_order])
ax2.set_ylabel("Mean Brier score")
ax2.set_title("Mean Brier by dispersion tercile")
ax2.legend(framealpha=0.9)

# Overall means annotation
ax2.axhline(df["bs_am"].mean(), color="#d62728", lw=0.8, ls="--")
ax2.axhline(df["bs_bc"].mean(), color="#1f77b4", lw=0.8, ls="--")

fig.suptitle(f"GJP Year 1: Brier score comparison ($n={len(df)}$ binary questions)",
             fontsize=11)
fig.tight_layout()
for ext, kw in [(".pdf", {}), (".png", {"dpi": 200})]:
    fig.savefig(FIG_DIR / f"fig_gjp1_brier{ext}", bbox_inches="tight", **kw)
    fig.savefig(GJP_DIR / f"fig_gjp1_brier{ext}", bbox_inches="tight", **kw)
plt.close(fig)
print("Saved fig_gjp1_brier")
