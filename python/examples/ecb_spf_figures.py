"""
Generates fig_ecb1_mass_correct_bin.pdf from output/ecb_spf_backtest/ecb_backtest.csv.

CSV columns: t, year, qtr, realized, bin_idx, n_forecasters, am_rps, bc_rps,
             am_log, bc_log, am_mass, bc_mass, dispersion, hi_disp
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

OUT_DIR = Path("output/ecb_spf_backtest")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

AM_COL = "#d62728"
BC_COL = "#1f77b4"

df = pd.read_csv(OUT_DIR / "ecb_backtest.csv")
print(f"ECB quarters: {len(df)}")

# ── fig_ecb1_mass_correct_bin ─────────────────────────────────────────────────
fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(12, 6), sharex=True,
    gridspec_kw={"height_ratios": [2, 1]},
)

# Shade high-dispersion quarters
for _, row in df[df["hi_disp"] == True].iterrows():
    ax_top.axvspan(row["t"] - 0.125, row["t"] + 0.125,
                   color="lightgrey", alpha=0.4, lw=0)
    ax_bot.axvspan(row["t"] - 0.125, row["t"] + 0.125,
                   color="lightgrey", alpha=0.4, lw=0)

# ── Top panel: probability on realized bin ────────────────────────────────────
ax_top.plot(df["t"], df["am_mass"], color=AM_COL, lw=1.4, alpha=0.85,
            label="Arithmetic mean")
ax_top.plot(df["t"], df["bc_mass"], color=BC_COL, lw=1.4, alpha=0.85,
            label="$W_1$ barycenter")

# Dashed mean lines
am_mean = df["am_mass"].mean()
bc_mean = df["bc_mass"].mean()
ax_top.axhline(am_mean, color=AM_COL, lw=0.8, ls="--")
ax_top.axhline(bc_mean, color=BC_COL, lw=0.8, ls="--")

# Right-side mean labels
t_end = df["t"].max()
ax_top.text(t_end + 0.05, bc_mean, f"{bc_mean:.3f}",
            color=BC_COL, fontsize=8, va="center")
ax_top.text(t_end + 0.05, am_mean, f"{am_mean:.3f}",
            color=AM_COL, fontsize=8, va="center")

ax_top.set_ylabel("Probability on realized bin")
ax_top.set_ylim(bottom=0)

# Legend with dispersion patch
disp_patch = mpatches.Patch(color="lightgrey", alpha=0.7, label="High dispersion")
ax_top.legend(handles=[
    plt.Line2D([0], [0], color=AM_COL, lw=1.4, label="Arithmetic mean"),
    plt.Line2D([0], [0], color=BC_COL, lw=1.4, label="$W_1$ barycenter"),
    disp_patch,
], loc="upper left", framealpha=0.9)

# Two-line title
ax_top.set_title(
    "ECB SPF: probability mass on realized HICP bin"
    " (2010Q1\u20132024Q3)\n"
    "(higher = better; shaded = high-dispersion quarters)",
    fontsize=10,
)

# ── Bottom panel: BC − AM difference ─────────────────────────────────────────
diff = df["bc_mass"] - df["am_mass"]
colors = [BC_COL if d >= 0 else AM_COL for d in diff]
bars = ax_bot.bar(df["t"], diff, width=0.22, color=colors, alpha=0.8)
ax_bot.axhline(0, color="black", lw=0.8)
ax_bot.set_ylabel("BC $-$ AM mass")
ax_bot.set_xlabel("Survey quarter")

# Bottom panel legend
bc_patch = mpatches.Patch(color=BC_COL, alpha=0.8, label="BC assigns more")
am_patch = mpatches.Patch(color=AM_COL, alpha=0.8, label="AM assigns more")
ax_bot.legend(handles=[bc_patch, am_patch], loc="upper left",
              framealpha=0.9, fontsize=8)

fig.tight_layout()
for ext, kw in [(".pdf", {}), (".png", {"dpi": 200})]:
    fig.savefig(FIG_DIR / f"fig_ecb1_mass_correct_bin{ext}",
                bbox_inches="tight", **kw)
    fig.savefig(OUT_DIR / f"fig_ecb1_mass_correct_bin{ext}",
                bbox_inches="tight", **kw)
plt.close(fig)
print("Saved fig_ecb1_mass_correct_bin")
