"""
Generates fig_gjp2_multicategory.pdf from output/gjp/gjp_multicategory_backtest.csv.

Scatter: BS(AM) vs BS(BC), colored by K (number of options).
Bar: mean BS by dispersion half and K.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

ROOT = Path(__file__).resolve().parents[2]
GJP_DIR = ROOT / "output" / "gjp"
FIG_DIR = ROOT.parent / "figures"  # repo root figures/
FIG_DIR.mkdir(exist_ok=True)

df = pd.read_csv(GJP_DIR / "gjp_multicategory_backtest.csv")
print(f"Multi-category questions: {len(df)}")

K_COLORS  = {3: "#1f77b4", 4: "#ff7f0e", 5: "#2ca02c"}
K_LABELS  = {3: "K=3", 4: "K=4", 5: "K=5"}

# ── subset: genuinely unordered questions ────────────────────────────────────
uo = df[df["cat"] == "unordered"].copy() if "cat" in df.columns else df.copy()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# ── Left: scatter AM vs BC Brier score, colored by K (all questions) ────────
for k, grp in df.groupby("K"):
    ax1.scatter(grp["bs_am"], grp["bs_bc"],
                color=K_COLORS[k], s=22, alpha=0.75, label=K_LABELS[k], zorder=3)
lo = 0
hi = max(df["bs_am"].max(), df["bs_bc"].max()) * 1.05
ax1.plot([lo, hi], [lo, hi], "k--", lw=0.9, label="BC = AM")
ax1.set_xlabel("AM Brier score")
ax1.set_ylabel("BC Brier score")
ax1.set_title(f"AM vs. BC Brier score ($n={len(df)}$ questions)")
ax1.legend(framealpha=0.9, loc="upper left")
ax1.set_xlim(lo, hi); ax1.set_ylim(lo, hi)

pct_below = (df["bs_bc"] < df["bs_am"]).mean() * 100
ax1.text(0.97, 0.05, f"BC wins {pct_below:.0f}%",
         ha="right", va="bottom", transform=ax1.transAxes, fontsize=9)

# ── Right: mean BS by K for genuinely unordered questions ────────────────────
ks = [3, 4, 5]
am_means = [uo[uo["K"] == k]["bs_am"].mean() for k in ks]
tm_means = [uo[uo["K"] == k]["bs_tm"].mean() for k in ks] if "bs_tm" in uo.columns else [None]*3
bc_means = [uo[uo["K"] == k]["bs_bc"].mean() for k in ks]
x = np.arange(len(ks))
w = 0.25
ax2.bar(x - w, am_means, width=w, color="#d62728", alpha=0.85, label="AM")
if tm_means[0] is not None:
    ax2.bar(x,      tm_means, width=w, color="#ff7f0e", alpha=0.85, label="TM (10%)")
ax2.bar(x + w, bc_means, width=w, color="#1f77b4", alpha=0.85, label="BC ($\\ell^1$)")
ax2.set_xticks(x)
ns = [len(uo[uo["K"]==k]) for k in ks]
ax2.set_xticklabels([f"K={k}\n(n={ns[i]})" for i, k in enumerate(ks)])
ax2.set_ylabel("Mean Brier score")
ax2.set_title(f"Mean Brier by $K$ (unordered questions, $n={len(uo)}$)")
ax2.legend(framealpha=0.9)

fig.suptitle(
    f"GJP Years 1–4: $\\ell^1$ barycenter vs. arithmetic mean and trimmed mean"
    f" ($n={len(df)}$ multi-category questions, K>2)",
    fontsize=11,
)
fig.tight_layout()
for ext, kw in [(".pdf", {}), (".png", {"dpi": 200})]:
    fig.savefig(FIG_DIR / f"fig_gjp2_multicategory{ext}", bbox_inches="tight", **kw)
plt.close(fig)
print("Saved fig_gjp2_multicategory")
