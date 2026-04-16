"""
Fig 5 (⭐ most important) — CDF fan chart: fixed BC vs. Q-Level in low-D and high-D quarters.
Shows that Q-Level visibly shifts the aggregate inside the panel geometry.
Adds bootstrap parameter-uncertainty bands around the Q-Level CDF.
Output: figures/fig_geometry.pdf
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
from scipy.optimize import minimize

# ── load ─────────────────────────────────────────────────────────────────────
from spf_backtest import load_all_quarters, BIN_UPPERS, cdf_median

alpha_hat, beta_hat = 0.65, -2.09

OOS_CSV    = Path(__file__).parent.parent.parent / "output" / "thread3" / "oos_results.csv"
TRAIN_CSV  = Path(__file__).parent.parent.parent / "output" / "spf_backtest" / "backtest_current.csv"

oos = pd.read_csv(OOS_CSV)
panel = load_all_quarters("current")   # dict: (yr, qt) -> pmfs array (N x K)

# pick the single most extreme low-D and high-D quarter from OOS
lo_row = oos.nsmallest(1, "disp").iloc[0]
hi_row = oos.nlargest(1, "disp").iloc[0]

# ── helper ───────────────────────────────────────────────────────────────────
def q_level_cdf(pmfs, q):
    """CDF-quantile aggregate at level q.  Returns (K-1,) CDF array."""
    cdfs = np.cumsum(pmfs, axis=1)[:, :-1]   # (N, K-1)
    return np.quantile(cdfs, q, axis=0)


def rps_from_cdf(cdf_agg, bin_idx):
    """RPS from aggregate CDF (K-1,) and realized bin index (0-based)."""
    K = len(cdf_agg) + 1
    cdf_r = np.zeros(K - 1)
    if bin_idx < K - 1:
        cdf_r[bin_idx:] = 1.0
    return float(np.sum((cdf_agg - cdf_r) ** 2))


# ── pre-compute training CDFs for fast bootstrap ──────────────────────────────
def build_training_arrays():
    """
    Load training data from backtest_current.csv (2007-2016, SA only).
    Pre-compute per-quarter CDF arrays for fast vectorised RPS evaluation.

    Returns:
        disp_arr  : (T,) array of cross-sectional dispersions
        bin_arr   : (T,) array of realized bin indices
        cdf_list  : list of (N_t, K-1) CDF arrays, length T
    """
    df = pd.read_csv(TRAIN_CSV)
    df = df[(df["series"] == "SA") & (df["year"] <= 2016)].copy()
    df = df.sort_values(["year", "qtr"]).reset_index(drop=True)

    disp_arr = df["dispersion"].values.astype(float)
    bin_arr  = df["bin_idx"].values.astype(int)
    cdf_list = []
    for _, row in df.iterrows():
        key = (int(row["year"]), int(row["qtr"]))
        pmfs = panel[key]   # (N_t, K)
        cdfs = np.cumsum(pmfs, axis=1)[:, :-1]   # (N_t, K-1)
        cdf_list.append(cdfs)

    return disp_arr, bin_arr, cdf_list


def total_rps_fast(params, disp_arr, bin_arr, cdf_list):
    """Vectorised total RPS for given (alpha, beta) on training data."""
    a, b = params
    qs = expit(a + b * disp_arr)        # (T,)
    total = 0.0
    for t, (q_t, bin_idx, cdfs_t) in enumerate(zip(qs, bin_arr, cdf_list)):
        cdf_agg = np.quantile(cdfs_t, q_t, axis=0)   # (K-1,)
        total += rps_from_cdf(cdf_agg, bin_idx)
    return total


def fit_qlevel_fast(boot_idx, disp_arr, bin_arr, cdf_list,
                    init_alpha=0.65, init_beta=-2.09):
    """Fit Q-Level on bootstrap-resampled training rows."""
    d_boot = disp_arr[boot_idx]
    b_boot = bin_arr[boot_idx]
    c_boot = [cdf_list[i] for i in boot_idx]
    res = minimize(total_rps_fast, x0=[init_alpha, init_beta],
                   args=(d_boot, b_boot, c_boot),
                   method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 2000})
    return res.x[0], res.x[1]


def bootstrap_qlevel_bands(oos_df, B=500, seed=42, conf=0.95):
    """
    Bootstrap training quarters (with replacement) → distribution of (α*, β*)
    → parameter-uncertainty bands on Ĝ_t(k) for each OOS quarter.

    Returns:
        bands       : dict (yr, qt) -> (lower, upper) each shape (K-1,)
        boot_params : (B, 2) array of (alpha*, beta*)
    """
    rng = np.random.default_rng(seed)
    disp_arr, bin_arr, cdf_list = build_training_arrays()
    T = len(disp_arr)
    print(f"  Training quarters: {T}")

    boot_params = []
    for b in range(B):
        idx = rng.integers(0, T, size=T)
        a_b, b_b = fit_qlevel_fast(idx, disp_arr, bin_arr, cdf_list,
                                    init_alpha=alpha_hat, init_beta=beta_hat)
        boot_params.append((a_b, b_b))
        if (b + 1) % 50 == 0:
            arr = np.array(boot_params)
            print(f"  Bootstrap {b+1}/{B}: "
                  f"α mean={arr[:,0].mean():.3f}, β mean={arr[:,1].mean():.3f}")

    boot_params = np.array(boot_params)  # (B, 2)

    alpha_lo = (1 - conf) / 2
    alpha_hi = 1 - alpha_lo
    bands = {}
    for _, row in oos_df.iterrows():
        yr, qt = int(row["year"]), int(row["qtr"])
        key = (yr, qt)
        if key not in panel:
            continue
        pmfs = panel[key]
        D = float(row["disp"])
        cdfs_t = np.cumsum(pmfs, axis=1)[:, :-1]

        boot_cdfs = np.array([
            np.quantile(cdfs_t, float(expit(a_b + b_b * D)), axis=0)
            for a_b, b_b in boot_params
        ])   # (B, K-1)

        bands[key] = (
            np.quantile(boot_cdfs, alpha_lo, axis=0),
            np.quantile(boot_cdfs, alpha_hi, axis=0),
        )

    return bands, boot_params


# ── compute bootstrap bands ───────────────────────────────────────────────────
print("Computing bootstrap parameter-uncertainty bands (B=500)...")
bands, boot_params = bootstrap_qlevel_bands(oos, B=500, seed=42)
print(f"  α* mean={boot_params[:,0].mean():.3f}  β* mean={boot_params[:,1].mean():.3f}")
print(f"  α* 95% CI: [{np.percentile(boot_params[:,0],2.5):.3f}, "
      f"{np.percentile(boot_params[:,0],97.5):.3f}]")
print(f"  β* 95% CI: [{np.percentile(boot_params[:,1],2.5):.3f}, "
      f"{np.percentile(boot_params[:,1],97.5):.3f}]")


# ── plot ──────────────────────────────────────────────────────────────────────
def plot_panel(ax, row, title):
    yr, qt = int(row["year"]), int(row["qtr"])
    D = float(row["disp"])
    q_theta = float(expit(alpha_hat + beta_hat * D))
    realized_bin = int(row["bin_idx"])

    pmfs = panel[(yr, qt)]
    K = pmfs.shape[1]
    bins = np.arange(K - 1)

    # individual CDFs (grey)
    cdfs_all = np.cumsum(pmfs, axis=1)[:, :-1]
    for cdf_i in cdfs_all:
        ax.step(bins, cdf_i, color="#BBBBBB", alpha=0.25, lw=0.6, where="post")

    # BC aggregate (q = 0.5)
    cdf_bc = q_level_cdf(pmfs, 0.5)
    ax.step(bins, cdf_bc, color="#2196F3", lw=2.5, where="post",
            label=f"BC ($q = 0.50$)", zorder=4)

    # Q-Level aggregate
    cdf_ql = q_level_cdf(pmfs, q_theta)
    ax.step(bins, cdf_ql, color="#E91E63", lw=2.5, where="post",
            label=f"Q-Level ($q = {q_theta:.2f}$)", zorder=5)

    # Bootstrap parameter-uncertainty band
    key = (yr, qt)
    if key in bands:
        lower, upper = bands[key]
        # step-interpolated fill
        x_step = np.concatenate([[bins[0] - 0.5],
                                  np.repeat(bins, 2)[1:],
                                  [bins[-1] + 0.5]])
        lo_step = np.repeat(lower, 2)
        hi_step = np.repeat(upper, 2)
        n = min(len(x_step), len(lo_step))
        ax.fill_between(x_step[:n], lo_step[:n], hi_step[:n],
                        color="#E91E63", alpha=0.18, zorder=3,
                        label=r"95% param.\ uncert.\ band for $\hat{G}_t(k)$")

    # realized bin
    ax.axvline(realized_bin - 0.5, color="black", lw=1.8, ls="--",
               label="Realized bin", zorder=6)

    ax.set_title(f"{title}\n{yr}:Q{qt},  $D_t = {D:.3f}$,  "
                 f"$n = {pmfs.shape[0]}$ forecasters", fontsize=9)
    ax.set_xlabel("Bin $k$", fontsize=9)
    ax.set_ylabel("CDF", fontsize=9)
    ax.set_xlim(-0.5, K - 1.5)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=8, loc="upper left")
    ax.tick_params(labelsize=8)


# ── figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
plot_panel(axes[0], lo_row, "Low-dispersion quarter")
plot_panel(axes[1], hi_row, "High-dispersion quarter")

fig.suptitle(
    "Q-Level shifts the aggregate inside the panel geometry\n"
    r"(grey: individual CDFs; blue: fixed BC at $q=0.5$; pink: learned Q-Level;"
    r" shading: 95\% parameter-uncertainty band for $\hat G_t(k)$)",
    fontsize=9.5, y=1.01
)
fig.tight_layout()

OUT_FIG = Path(__file__).parent.parent.parent.parent / "figures" / "fig_geometry.pdf"
fig.savefig(OUT_FIG, bbox_inches="tight")
print(f"Saved {OUT_FIG}")
print(f"  Low-D:  {int(lo_row['year'])}:Q{int(lo_row['qtr'])}, "
      f"D={lo_row['disp']:.3f}, q_θ={expit(alpha_hat + beta_hat*lo_row['disp']):.3f}")
print(f"  High-D: {int(hi_row['year'])}:Q{int(hi_row['qtr'])}, "
      f"D={hi_row['disp']:.3f}, q_θ={expit(alpha_hat + beta_hat*hi_row['disp']):.3f}")
