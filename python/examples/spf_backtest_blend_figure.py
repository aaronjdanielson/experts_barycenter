"""
Regenerates fig_bt7_blend.pdf and fig_bt6_mass_correct_bin.pdf from saved CSVs.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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

OUT_DIR  = Path("output/spf_backtest")
FIG_DIR  = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

# ── Colours ──────────────────────────────────────────────────────────────────
AM_COL     = "#d62728"
BC_COL     = "#1f77b4"
FIXED_COL  = "#2ca02c"
ADAPT_COL  = "#ff7f0e"

# ─────────────────────────────────────────────────────────────────────────────
# fig_bt7_blend: left = oracle RPS vs lambda; right = cumulative OOS mean RPS
# ─────────────────────────────────────────────────────────────────────────────
df_bt   = pd.read_csv(OUT_DIR / "backtest_current.csv")
df_bt   = df_bt[df_bt["series"] == "NSA"].reset_index(drop=True)
oos     = pd.read_csv(OUT_DIR / "blend_oos.csv").sort_values("t").reset_index(drop=True)

# Left panel: oracle RPS vs lambda
lambdas  = np.linspace(0, 1, 201)
am_arr   = df_bt["am_rps"].values
bc_arr   = df_bt["bc_rps"].values
rps_grid = [np.mean(lam * bc_arr + (1 - lam) * am_arr) for lam in lambdas]
lam_star = lambdas[np.argmin(rps_grid)]

# Right panel: cumulative (expanding) mean RPS over OOS period
def cum_mean(series):
    return np.cumsum(series) / np.arange(1, len(series) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# — Left panel —
ax1.plot(lambdas, rps_grid, color="#333333", lw=1.8)
ax1.axvline(lam_star, color="grey", lw=1, ls="--")
ax1.scatter([0, 1], [rps_grid[0], rps_grid[-1]], s=60,
            color=[AM_COL, BC_COL], zorder=5)
# AM label: upper-left
ax1.annotate(f"AM ($\\lambda=0$)\nRPS={rps_grid[0]:.3f}",
             xy=(0, rps_grid[0]), xytext=(0.06, rps_grid[0] + 0.0005),
             fontsize=8, color=AM_COL)
# BC label: avoid overlapping the arrow when lam_star==1
bc_txt_x = 0.60 if lam_star >= 0.95 else 0.72
ax1.annotate(f"BC ($\\lambda=1$)\nRPS={rps_grid[-1]:.3f}",
             xy=(1, rps_grid[-1]), xytext=(bc_txt_x, rps_grid[-1] + 0.0005),
             fontsize=8, color=BC_COL)
# lam* arrow — offset left when at boundary
arr_dx = -0.08 if lam_star >= 0.95 else 0.05
ax1.annotate(f"$\\lambda^*={lam_star:.2f}$",
             xy=(lam_star, min(rps_grid)),
             xytext=(lam_star + arr_dx, min(rps_grid) + 0.001),
             fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
ax1.set_xlabel("Blend weight $\\lambda$")
ax1.set_ylabel("Mean RPS (full sample)")
ax1.set_title("Oracle blend weight")
ax1.set_xlim(-0.02, 1.02)
ax1.grid(True, alpha=0.25, lw=0.6)

# — Right panel —
t_vals = oos["t"].values
series = [
    ("rps_am",       AM_COL,    1.8, "-",  "AM"),
    ("rps_bc",       BC_COL,    1.8, "-",  "Barycenter"),
    ("rps_fixed",    FIXED_COL, 1.6, "--", "Fixed blend"),
    ("rps_adaptive", ADAPT_COL, 2.2, "-",  "Adaptive blend"),
]
cum_vals = {col: cum_mean(oos[col]) for col, *_ in series}
for col, color, lw, ls, _ in series:
    ax2.plot(t_vals, cum_vals[col], color=color, lw=lw, ls=ls)

# End-of-line labels: extend x-axis right for label room
t_end   = t_vals[-1]
x_right = t_end + 2.2
ax2.set_xlim(t_vals[0] - 0.3, x_right)

# Auto-space labels: sort by end value, enforce min gap of 0.007
MIN_GAP = 0.007
label_info = [(col, color, name, cum_vals[col].iloc[-1])
              for col, color, _, _, name in series]
label_info.sort(key=lambda x: x[3])          # sort ascending by end value
# Push labels apart bottom-up
y_placed = []
for col, color, name, y_raw in label_info:
    y = y_raw
    for yp in y_placed:
        if abs(y - yp) < MIN_GAP:
            y = yp + MIN_GAP
    y_placed.append(y)
    ax2.annotate(name, xy=(t_end, y_raw),
                 xytext=(t_end + 0.15, y),
                 color=color, fontsize=8.5, va="center", fontweight="bold",
                 annotation_clip=False)

ax2.set_xlabel("Survey quarter")
ax2.set_ylabel("Cumulative mean RPS")
ax2.set_title("Out-of-sample evaluation (56 quarters)")
ax2.grid(True, alpha=0.25, lw=0.6)

fig.tight_layout()
for ext, kw in [(".pdf", {}), (".png", {"dpi": 200})]:
    fig.savefig(FIG_DIR / f"fig_bt7_blend{ext}", bbox_inches="tight", **kw)
    fig.savefig(OUT_DIR / f"fig_bt7_blend{ext}", bbox_inches="tight", **kw)
plt.close(fig)
print("Saved fig_bt7_blend")

# ─────────────────────────────────────────────────────────────────────────────
# fig_bt6_mass_correct_bin: top = per-quarter mass; bottom = BC - AM diff
# ─────────────────────────────────────────────────────────────────────────────
df = df_bt.copy()
df["t"] = df["year"] + (df["qtr"] - 1) / 4
med_disp = df["dispersion"].median()
df["hi_disp"] = df["dispersion"] >= med_disp

fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                                      gridspec_kw={"height_ratios": [2, 1]})

# Shade high-dispersion quarters
for _, row in df[df["hi_disp"]].iterrows():
    ax_top.axvspan(row["t"] - 0.125, row["t"] + 0.125, color="lightgrey", alpha=0.4, lw=0)
    ax_bot.axvspan(row["t"] - 0.125, row["t"] + 0.125, color="lightgrey", alpha=0.4, lw=0)

ax_top.plot(df["t"], df["am_mass"], color=AM_COL, lw=1.4, alpha=0.85, label="Arith. mean")
ax_top.plot(df["t"], df["bc_mass"], color=BC_COL, lw=1.4, alpha=0.85, label="$W_1$ barycenter")
ax_top.axhline(df["am_mass"].mean(), color=AM_COL, lw=0.8, ls="--")
ax_top.axhline(df["bc_mass"].mean(), color=BC_COL, lw=0.8, ls="--")
ax_top.annotate(f"AM mean: {df['am_mass'].mean():.3f}",
                xy=(df["t"].iloc[-1], df["am_mass"].mean()),
                xytext=(-5, -12), textcoords="offset points",
                fontsize=8, color=AM_COL, ha="right")
ax_top.annotate(f"BC mean: {df['bc_mass'].mean():.3f}",
                xy=(df["t"].iloc[-1], df["bc_mass"].mean()),
                xytext=(-5, 4), textcoords="offset points",
                fontsize=8, color=BC_COL, ha="right")
ax_top.set_ylabel("P(realized bin)")
ax_top.legend(loc="upper left", framealpha=0.9)

diff = df["bc_mass"] - df["am_mass"]
colors = [BC_COL if d >= 0 else AM_COL for d in diff]
ax_bot.bar(df["t"], diff, width=0.22, color=colors, alpha=0.8)
ax_bot.axhline(0, color="black", lw=0.8)
ax_bot.set_ylabel("BC $-$ AM")
ax_bot.set_xlabel("Survey quarter")

fig.suptitle("Probability mass on realized bin: AM vs.\ $W_1$ barycenter (2007--2025)",
             fontsize=11)
fig.tight_layout()
for ext, kw in [(".pdf", {}), (".png", {"dpi": 200})]:
    fig.savefig(FIG_DIR / f"fig_bt6_mass_correct_bin{ext}", bbox_inches="tight", **kw)
    fig.savefig(OUT_DIR / f"fig_bt6_mass_correct_bin{ext}", bbox_inches="tight", **kw)
plt.close(fig)
print("Saved fig_bt6_mass_correct_bin")
