"""
High-quality figures for the SPF historical backtest.

Produces:
  fig_bt1_scores_timeseries.pdf  — AM vs BC RPS over time, dispersion shaded
  fig_bt2_score_diff.pdf         — Quarter-by-quarter BC−AM RPS difference
  fig_bt3_dispersion_split.pdf   — Mean RPS by method × dispersion group
  fig_bt4_scatter.pdf            — AM vs BC RPS per quarter scatter

Interpreter: /Users/aarondanielson/miniforge3/envs/supercluster/bin/python3.11
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
from pathlib import Path

ROOT    = Path(__file__).parent.parent.parent
OUT_DIR = ROOT / "output" / "spf_backtest"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.linewidth":   0.8,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "pdf.fonttype":     42,       # embeds fonts for journal submission
    "ps.fonttype":      42,
})

AM_COLOR = "#c0392b"
BC_COLOR = "#2471a3"
HI_ALPHA = 0.13
LO_ALPHA = 0.0

def quarter_float(year, qtr):
    return year + (qtr - 1) / 4


# ── Load data ─────────────────────────────────────────────────────────────────

df = pd.read_csv(OUT_DIR / "backtest_current.csv")
df = df[df["series"] == "SA"].copy().reset_index(drop=True)
df["t"] = df.apply(lambda r: quarter_float(r["year"], r["qtr"]), axis=1)
med_disp = df["dispersion"].median()
df["hi_disp"] = df["dispersion"] >= med_disp
df["rps_diff"] = df["bc_rps"] - df["am_rps"]   # negative = BC wins


# ── Fig 1: Time series of RPS with dispersion shading ─────────────────────────

fig, ax = plt.subplots(figsize=(9, 4))

# Shade high-dispersion quarters
in_band = False
band_start = None
for _, row in df.sort_values("t").iterrows():
    if row["hi_disp"] and not in_band:
        band_start = row["t"] - 0.125
        in_band = True
    elif not row["hi_disp"] and in_band:
        ax.axvspan(band_start, row["t"] - 0.125, color="gray", alpha=HI_ALPHA, lw=0)
        in_band = False
if in_band:
    ax.axvspan(band_start, df["t"].max() + 0.125, color="gray", alpha=HI_ALPHA, lw=0)

ax.plot(df["t"], df["am_rps"], color=AM_COLOR, lw=1.6, label="Arithmetic mean", zorder=3)
ax.plot(df["t"], df["bc_rps"], color=BC_COLOR, lw=1.6, label="W$_1$ barycenter", zorder=3)

ax.set_xlabel("Survey quarter")
ax.set_ylabel("Ranked Probability Score")
ax.set_title("SPF core CPI forecasts: RPS over time\n(shaded = high-dispersion quarters)",
             fontsize=11, pad=8)
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.set_xlim(df["t"].min() - 0.3, df["t"].max() + 0.3)

hi_patch = mpatches.Patch(color="gray", alpha=0.3, label="High dispersion")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles + [hi_patch], labels + ["High dispersion"],
          fontsize=9, frameon=False, loc="upper left")

fig.tight_layout()
fig.savefig(OUT_DIR / "fig_bt1_scores_timeseries.pdf", bbox_inches="tight")
fig.savefig(OUT_DIR / "fig_bt1_scores_timeseries.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved fig_bt1_scores_timeseries")


# ── Fig 2: Quarter-by-quarter BC − AM RPS difference ─────────────────────────

fig, ax = plt.subplots(figsize=(9, 3.5))

df_s = df.sort_values("t").reset_index(drop=True)
colors = [BC_COLOR if v < 0 else AM_COLOR for v in df_s["rps_diff"]]
ax.bar(df_s["t"], df_s["rps_diff"], width=0.22, color=colors, alpha=0.85, zorder=3)
ax.axhline(0, color="black", lw=0.8, zorder=2)

# Shade high-dispersion quarters (reuse same logic)
in_band = False
band_start = None
for _, row in df_s.iterrows():
    if row["hi_disp"] and not in_band:
        band_start = row["t"] - 0.125
        in_band = True
    elif not row["hi_disp"] and in_band:
        ax.axvspan(band_start, row["t"] - 0.125, color="gray", alpha=HI_ALPHA, lw=0)
        in_band = False
if in_band:
    ax.axvspan(band_start, df_s["t"].max() + 0.125, color="gray", alpha=HI_ALPHA, lw=0)

ax.set_xlabel("Survey quarter")
ax.set_ylabel("BC $-$ AM ranked probability score")
ax.set_title("Score difference per quarter (negative = barycenter wins)\n"
             "(shaded = high-dispersion quarters)", fontsize=11, pad=8)
ax.set_xlim(df_s["t"].min() - 0.3, df_s["t"].max() + 0.3)

bc_patch = mpatches.Patch(color=BC_COLOR, alpha=0.85, label="BC wins")
am_patch = mpatches.Patch(color=AM_COLOR, alpha=0.85, label="AM wins")
hi_patch = mpatches.Patch(color="gray",   alpha=0.3,  label="High dispersion")
ax.legend(handles=[bc_patch, am_patch, hi_patch], fontsize=9, frameon=False)

fig.tight_layout()
fig.savefig(OUT_DIR / "fig_bt2_score_diff.pdf", bbox_inches="tight")
fig.savefig(OUT_DIR / "fig_bt2_score_diff.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved fig_bt2_score_diff")


# ── Fig 3: Dispersion-split bar chart ─────────────────────────────────────────

hi = df[df["hi_disp"]]
lo = df[~df["hi_disp"]]

fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=False)

for ax, sub, label in zip(axes, [lo, hi], ["Low dispersion", "High dispersion"]):
    vals = [sub["am_rps"].mean(), sub["bc_rps"].mean()]
    bars = ax.bar(["Arith.\nmean", "W$_1$\nbarycenter"], vals,
                  color=[AM_COLOR, BC_COLOR], alpha=0.85, width=0.5)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=10)
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_title(f"{label}\n(n={len(sub)} quarters)", fontsize=11)
    ax.set_ylabel("Mean RPS" if ax is axes[0] else "")
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))

fig.suptitle("Mean RPS by aggregation method and panel dispersion\n"
             "(lower RPS = better forecast accuracy)", fontsize=11, y=1.01)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig_bt3_dispersion_split.pdf", bbox_inches="tight")
fig.savefig(OUT_DIR / "fig_bt3_dispersion_split.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved fig_bt3_dispersion_split")


# ── Fig 4: AM vs BC scatter ───────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(5, 5))

hi_mask = df["hi_disp"].values
ax.scatter(df.loc[~hi_mask, "am_rps"], df.loc[~hi_mask, "bc_rps"],
           c=BC_COLOR, alpha=0.6, s=30, label="Low dispersion", zorder=3)
ax.scatter(df.loc[hi_mask, "am_rps"], df.loc[hi_mask, "bc_rps"],
           c="gray", alpha=0.5, s=30, label="High dispersion", zorder=3)

lims = [0, max(df["am_rps"].max(), df["bc_rps"].max()) * 1.05]
ax.plot(lims, lims, "k--", lw=0.9, alpha=0.5, label="AM = BC")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("AM ranked probability score")
ax.set_ylabel("W$_1$ barycenter ranked probability score")
ax.set_title("AM vs.\ barycenter RPS per quarter\n(below diagonal = barycenter wins)", fontsize=11)
ax.legend(fontsize=9, frameon=False)

fig.tight_layout()
fig.savefig(OUT_DIR / "fig_bt4_scatter.pdf", bbox_inches="tight")
fig.savefig(OUT_DIR / "fig_bt4_scatter.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved fig_bt4_scatter")

print(f"\nAll figures saved to {OUT_DIR}")
