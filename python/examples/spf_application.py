"""
Application: Survey of Professional Forecasters (SPF) – Core CPI Probability Forecasts.

Data: Philadelphia Fed SPF microdata. Each participating forecaster assigns probability
mass to 10 ordered bins for the fourth-quarter-over-fourth-quarter growth rate of core CPI
(PRCCPI). Bins (descending), from SPF Table 8 documentation:
  1: ≥ 4.0%,  2: 3.5–3.9%,  3: 3.0–3.4%,  4: 2.5–2.9%,  5: 2.0–2.4%
  6: 1.5–1.9%,  7: 1.0–1.4%,  8: 0.5–0.9%,  9: 0.0–0.4%,  10: will decline

We re-index in ascending order (bin 1 → "will decline") for the analysis.

Survey: 2022 Q1 (filed Feb 2022, peak-inflation era). We use the NEXT-YEAR histograms
(columns PRCCPI11–PRCCPI20 = forecast for 2023 core CPI). This is a high-uncertainty
period with wide disagreement across forecasters.

Produces figures for Section 7 (SPF Application) of the paper.
"""

import numpy as np
import openpyxl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from wbarycenter import dw_barycenter

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "spf" / "SPFmicrodata.xlsx"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "spf"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Survey quarter to use
YEAR, QTR = 2022.0, 1.0

# Bin labels in ASCENDING inflation order (we reverse the SPF encoding)
BIN_LABELS = [
    "Will\ndecline",
    "0.0–0.4",
    "0.5–0.9",
    "1.0–1.4",
    "1.5–1.9",
    "2.0–2.4",
    "2.5–2.9",
    "3.0–3.4",
    "3.5–3.9",
    "≥ 4.0",
]
K = len(BIN_LABELS)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_spf_data():
    """Return (data, n_experts) for PRCCPI next-year forecasts in the chosen quarter.

    data: (n, K) array, rows are experts, columns are probability masses (sum to 1),
          bins in ascending inflation order.
    """
    wb = openpyxl.load_workbook(DATA_PATH, read_only=True)
    ws = wb["PRCCPI"]

    header = None
    quarter_rows = []
    for row in ws.iter_rows(values_only=True):
        if header is None:
            header = row
            continue
        if row[0] == YEAR and row[1] == QTR:
            quarter_rows.append(row)

    prob_cols = [c for c in header if c and str(c).startswith("PRCCPI")]
    nxt_cols = prob_cols[10:20]  # PRCCPI11–PRCCPI20 (next-year forecast)

    valid = []
    for r in quarter_rows:
        vals = [
            0.0 if (r[header.index(c)] == "#N/A" or r[header.index(c)] is None)
            else float(r[header.index(c)])
            for c in nxt_cols
        ]
        s = sum(vals)
        if 95.0 <= s <= 105.0:
            # normalise to 1, reverse to ascending order
            arr = np.array(vals) / s
            valid.append(arr[::-1])

    return np.array(valid)


# ---------------------------------------------------------------------------
# W1 distance matrix for ordered bins (absolute rank difference)
# ---------------------------------------------------------------------------

def w1_distance_matrix(k):
    """k×k matrix with D[i,j] = |i-j| (rank distance for uniform spacing)."""
    idx = np.arange(k)
    return np.abs(idx[:, None] - idx[None, :]).astype(float)


# ---------------------------------------------------------------------------
# CDF median (closed-form W1 barycenter for ordered outcomes)
# ---------------------------------------------------------------------------

def cdf_median(data):
    """Component-wise median of the CDFs, then recover PMF by differencing."""
    cdfs = np.cumsum(data, axis=1)          # (n, K)
    cdf_med = np.median(cdfs, axis=0)       # (K,)  -- component-wise median
    pmf = np.diff(np.concatenate([[0.0], cdf_med]))
    pmf = np.maximum(pmf, 0.0)
    pmf /= pmf.sum()
    return pmf


# ---------------------------------------------------------------------------
# Figure 1: Heatmap of all expert histograms
# ---------------------------------------------------------------------------

def fig_heatmap(data):
    n, K = data.shape
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(data * 100, aspect="auto", cmap="YlOrRd", vmin=0, vmax=70)
    ax.set_xticks(range(K))
    ax.set_xticklabels(BIN_LABELS, fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"Forecaster {i+1}" for i in range(n)], fontsize=7)
    plt.colorbar(im, ax=ax, label="Probability (%)")
    ax.set_title(
        f"SPF 2022:Q1 — Core CPI probability forecasts for 2023\n"
        f"({n} forecasters, 10 ordered bins)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_spf1_heatmap.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig_spf1_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_spf1_heatmap")


# ---------------------------------------------------------------------------
# Figure 2: AM vs W1-Barycenter (= CDF median) aggregate distributions
# ---------------------------------------------------------------------------

def fig_aggregate_comparison(am, bc_lp, bc_opt):
    x = np.arange(K)
    width = 0.28

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, am * 100, width, label="Arithmetic mean", color="#d62728", alpha=0.8)
    ax.bar(x,         bc_lp * 100, width, label="W1 barycenter (CDF median)", color="#1f77b4", alpha=0.8)
    ax.bar(x + width, bc_opt * 100, width, label="W1 barycenter (LP solver)", color="#2ca02c", alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS, fontsize=8)
    ax.set_xlabel("Core CPI bin (4th-qtr-over-4th-qtr change, %)")
    ax.set_ylabel("Probability (%)")
    ax.set_title(
        "SPF 2022:Q1: Aggregated 2023 core CPI forecasts\n"
        "Arithmetic mean vs. Wasserstein barycenter",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_spf2_aggregates.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig_spf2_aggregates.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_spf2_aggregates")


# ---------------------------------------------------------------------------
# Figure 3: Individual CDFs + aggregate CDFs
# ---------------------------------------------------------------------------

def fig_cdf_comparison(data, am, bc_lp):
    n = data.shape[0]
    x = np.arange(K)

    fig, ax = plt.subplots(figsize=(9, 5))

    # individual CDFs (thin, grey)
    for i in range(n):
        cdf_i = np.cumsum(data[i]) * 100
        ax.step(x, cdf_i, where="post", color="gray", alpha=0.25, lw=0.8)

    # AM CDF
    cdf_am = np.cumsum(am) * 100
    ax.step(x, cdf_am, where="post", color="#d62728", lw=2.5,
            label="Arithmetic mean CDF")

    # BC CDF (= CDF median)
    cdf_bc = np.cumsum(bc_lp) * 100
    ax.step(x, cdf_bc, where="post", color="#1f77b4", lw=2.5,
            label="Wasserstein barycenter CDF\n(= component-wise CDF median)")

    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS, fontsize=8)
    ax.set_ylabel("Cumulative probability (%)")
    ax.set_xlabel("Core CPI bin (4th-qtr-over-4th-qtr change, %)")
    ax.set_title(
        "SPF 2022:Q1: Individual and aggregate CDFs for 2023 core CPI",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_spf3_cdfs.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig_spf3_cdfs.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_spf3_cdfs")


# ---------------------------------------------------------------------------
# Figure 4: Leave-one-out influence — AM vs BC shift
# ---------------------------------------------------------------------------

def fig_leave_one_out(data):
    n, K = data.shape
    D = w1_distance_matrix(K)
    weights = np.ones(n) / n

    am_full = data.mean(axis=0)
    bc_full = cdf_median(data)

    am_shifts, bc_shifts = [], []
    print("  Computing leave-one-out (n runs)...")
    for i in range(n):
        dat_loo = np.delete(data, i, axis=0)
        am_loo = dat_loo.mean(axis=0)
        bc_loo = cdf_median(dat_loo)
        am_shifts.append(np.sum(np.abs(am_loo - am_full)))
        bc_shifts.append(np.sum(np.abs(bc_loo - bc_full)))

    am_shifts = np.array(am_shifts)
    bc_shifts = np.array(bc_shifts)
    order = np.argsort(am_shifts)[::-1]

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(n)
    ax.bar(x - 0.2, am_shifts[order] * 100, 0.38,
           label="Arithmetic Mean", color="#d62728", alpha=0.8)
    ax.bar(x + 0.2, bc_shifts[order] * 100, 0.38,
           label="Wasserstein Barycenter", color="#1f77b4", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"F{order[i]+1}" for i in range(n)],
                       rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(r"$\ell^1$ shift in aggregate (pp) when forecaster removed")
    ax.set_title("SPF 2022:Q1: Leave-one-out influence of each forecaster")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_spf4_loo.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig_spf4_loo.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_spf4_loo")
    return am_shifts, bc_shifts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Loading SPF data (PRCCPI next-year, {int(YEAR)}Q{int(QTR)})...")
    data = load_spf_data()
    n, K = data.shape
    print(f"  {n} forecasters, {K} bins")

    print("Computing aggregates...")
    weights = np.ones(n) / n
    D = w1_distance_matrix(K)

    am = data.mean(axis=0)
    bc_lp = cdf_median(data)               # closed-form W1 barycenter
    bc_opt, _ = dw_barycenter(data, weights, D)   # LP verification

    print(f"\n  AM  (bin probs %): {np.round(am * 100, 1)}")
    print(f"  BC  (CDF median %): {np.round(bc_lp * 100, 1)}")
    print(f"  BC  (LP solver %): {np.round(bc_opt * 100, 1)}")
    print(f"  |BC_lp - BC_opt| L1: {np.sum(np.abs(bc_lp - bc_opt)):.4f}")

    print("\n  AM vs BC per bin:")
    print(f"  {'Bin':>10}  {'AM%':>6}  {'BC%':>6}  {'diff':>7}")
    for i, lbl in enumerate(BIN_LABELS):
        print(f"  {lbl:>10}  {am[i]*100:>6.1f}  {bc_lp[i]*100:>6.1f}  {(am[i]-bc_lp[i])*100:>+7.1f}")

    print("\nGenerating figures...")
    fig_heatmap(data)
    fig_aggregate_comparison(am, bc_lp, bc_opt)
    fig_cdf_comparison(data, am, bc_lp)
    am_shifts, bc_shifts = fig_leave_one_out(data)

    print(f"\n  Mean LOO AM shift: {am_shifts.mean()*100:.2f}pp")
    print(f"  Mean LOO BC shift: {bc_shifts.mean()*100:.2f}pp")
    print(f"  Ratio BC/AM: {bc_shifts.mean()/am_shifts.mean():.3f}")

    print(f"\nAll figures saved to {OUTPUT_DIR}")
