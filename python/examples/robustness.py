"""
Robustness analysis: Arithmetic Mean vs. Wasserstein Barycenter.

Produces all figures for the Robustness section of the paper.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from wbarycenter import dw_barycenter

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "robustness"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STATE_LABELS = ["S1", "S2", "S3", "S4", "S5", "S6"]
K = 6


# ---------------------------------------------------------------------------
# Core simulation helpers
# ---------------------------------------------------------------------------

def _run_once(alpha_regular, n_regular, alpha_outlier, n_outlier, rng):
    n_total = n_regular + n_outlier
    weights = np.ones(n_total) / n_total
    D = np.ones((K, K)) - np.eye(K)

    dat = np.vstack([
        rng.dirichlet(alpha_regular, size=n_regular),
        rng.dirichlet(alpha_outlier, size=n_outlier),
    ])
    amean = dat.mean(axis=0)
    bcenter, _ = dw_barycenter(dat, weights, D)
    return amean, bcenter, dat


def run_simulation(R=200, n_regular=10, n_outlier=1, seed=42):
    rng = np.random.default_rng(seed)
    alpha_regular = np.ones(K)
    alpha_outlier = np.array([1, 1, 1, 1, 1.5, 10.0]) ** 4

    ameans = np.zeros((R, K))
    bmeans = np.zeros((R, K))
    for r in range(R):
        am, bc, _ = _run_once(alpha_regular, n_regular, alpha_outlier, n_outlier, rng)
        ameans[r] = am
        bmeans[r] = bc
    return ameans, bmeans


# ---------------------------------------------------------------------------
# Figure 1: Scatter — AM vs BC on each state across R replicates
# ---------------------------------------------------------------------------

def fig_scatter(ameans, bmeans):
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    for k, ax in enumerate(axes.flat):
        ax.scatter(ameans[:, k], bmeans[:, k], alpha=0.4, s=12, color="steelblue")
        lo = min(ameans[:, k].min(), bmeans[:, k].min()) - 0.01
        hi = max(ameans[:, k].max(), bmeans[:, k].max()) + 0.01
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="AM = BC")
        ax.set_xlabel("Arithmetic Mean")
        ax.set_ylabel("Barycenter")
        ax.set_title(f"State {STATE_LABELS[k]}")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("AM vs. Barycenter by State (each point = one simulation draw)",
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig1_scatter_am_vs_bc.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig1_scatter_am_vs_bc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig1_scatter_am_vs_bc")


# ---------------------------------------------------------------------------
# Figure 2: Boxplots — distribution of AM and BC on the outlier state
# ---------------------------------------------------------------------------

def fig_outlier_boxplot(ameans, bmeans, outlier_state=5):
    fig, ax = plt.subplots(figsize=(6, 4))
    data_to_plot = [ameans[:, outlier_state], bmeans[:, outlier_state]]
    bp = ax.boxplot(data_to_plot, tick_labels=["Arithmetic\nMean", "Wasserstein\nBarycenter"],
                    patch_artist=True, widths=0.5)
    colors = ["#d62728", "#1f77b4"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel(f"Probability assigned to {STATE_LABELS[outlier_state]}")
    ax.set_title("Effect of one outlying expert on the aggregate\n"
                 f"(outlier concentrates mass on {STATE_LABELS[outlier_state]})")
    ax.axhline(1 / K, color="gray", linestyle=":", lw=1, label="Uniform (1/K)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig2_outlier_boxplot.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig2_outlier_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig2_outlier_boxplot")


# ---------------------------------------------------------------------------
# Figure 3: Variance comparison across all states
# ---------------------------------------------------------------------------

def fig_variance_comparison(ameans, bmeans):
    am_var = ameans.var(axis=0)
    bc_var = bmeans.var(axis=0)
    x = np.arange(K)
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, am_var, width, label="Arithmetic Mean",
           color="#d62728", alpha=0.75)
    ax.bar(x + width / 2, bc_var, width, label="Wasserstein Barycenter",
           color="#1f77b4", alpha=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels(STATE_LABELS)
    ax.set_ylabel("Variance across simulation replicates")
    ax.set_title("Variance of AM vs. Barycenter by State")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig3_variance_comparison.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig3_variance_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig3_variance_comparison")


# ---------------------------------------------------------------------------
# Figure 4: Single-draw example — expert opinions + AM + BC as bar charts
# ---------------------------------------------------------------------------

def fig_single_draw_example(seed=0):
    rng = np.random.default_rng(seed)
    alpha_regular = np.ones(K)
    alpha_outlier = np.array([1, 1, 1, 1, 1.5, 10.0]) ** 4
    n_regular, n_outlier = 5, 1

    am, bc, dat = _run_once(alpha_regular, n_regular, alpha_outlier, n_outlier, rng)
    n_total = n_regular + n_outlier
    x = np.arange(K)

    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(1, n_total + 2, figure=fig, wspace=0.4)

    # Individual experts
    for i in range(n_total):
        ax = fig.add_subplot(gs[0, i])
        color = "#d95f02" if i == n_total - 1 else "#7570b3"
        ax.bar(x, dat[i], color=color, alpha=0.8)
        ax.set_ylim(0, 1)
        ax.set_xticks(x)
        ax.set_xticklabels(STATE_LABELS, fontsize=7)
        ax.set_title(f"Expert {i+1}" + (" (outlier)" if i == n_total - 1 else ""),
                     fontsize=8)
        if i == 0:
            ax.set_ylabel("Probability")

    # AM
    ax_am = fig.add_subplot(gs[0, n_total])
    ax_am.bar(x, am, color="#d62728", alpha=0.85)
    ax_am.set_ylim(0, 1)
    ax_am.set_xticks(x)
    ax_am.set_xticklabels(STATE_LABELS, fontsize=7)
    ax_am.set_title("Arith.\nMean", fontsize=9)

    # BC
    ax_bc = fig.add_subplot(gs[0, n_total + 1])
    ax_bc.bar(x, bc, color="#1f77b4", alpha=0.85)
    ax_bc.set_ylim(0, 1)
    ax_bc.set_xticks(x)
    ax_bc.set_xticklabels(STATE_LABELS, fontsize=7)
    ax_bc.set_title("Barycenter", fontsize=9)

    fig.suptitle("Expert opinions, Arithmetic Mean, and Wasserstein Barycenter",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig4_single_draw.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig4_single_draw.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig4_single_draw")


# ---------------------------------------------------------------------------
# Figure 5: Sensitivity — vary number of outliers
# ---------------------------------------------------------------------------

def fig_sensitivity_n_outliers(R=200, seed=42):
    rng = np.random.default_rng(seed)
    alpha_regular = np.ones(K)
    alpha_outlier = np.array([1, 1, 1, 1, 1.5, 10.0]) ** 4
    outlier_state = 5
    n_regular = 10
    outlier_counts = [0, 1, 2, 3]

    am_means_by_n = []
    bc_means_by_n = []

    for n_outlier in outlier_counts:
        n_total = n_regular + n_outlier
        weights = np.ones(n_total) / n_total
        D = np.ones((K, K)) - np.eye(K)
        ams, bcs = [], []
        for _ in range(R):
            dat = np.vstack([
                rng.dirichlet(alpha_regular, size=n_regular),
                *(rng.dirichlet(alpha_outlier, size=1) for _ in range(n_outlier)),
            ])
            ams.append(dat.mean(axis=0)[outlier_state])
            bc, _ = dw_barycenter(dat, weights, D)
            bcs.append(bc[outlier_state])
        am_means_by_n.append(np.array(ams))
        bc_means_by_n.append(np.array(bcs))

    fig, ax = plt.subplots(figsize=(7, 4))
    positions_am = np.array(outlier_counts) - 0.15
    positions_bc = np.array(outlier_counts) + 0.15
    bp1 = ax.boxplot(am_means_by_n, positions=positions_am, widths=0.25,
                     patch_artist=True, manage_ticks=False)
    bp2 = ax.boxplot(bc_means_by_n, positions=positions_bc, widths=0.25,
                     patch_artist=True, manage_ticks=False)
    for patch in bp1["boxes"]:
        patch.set_facecolor("#d62728"); patch.set_alpha(0.6)
    for patch in bp2["boxes"]:
        patch.set_facecolor("#1f77b4"); patch.set_alpha(0.6)
    ax.set_xticks(outlier_counts)
    ax.set_xticklabels([str(n) for n in outlier_counts])
    ax.set_xlabel("Number of outlying experts")
    ax.set_ylabel(f"Probability on {STATE_LABELS[outlier_state]}")
    ax.set_title("Sensitivity to number of outliers")
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]],
              ["Arithmetic Mean", "Wasserstein Barycenter"], fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig5_sensitivity_outliers.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig5_sensitivity_outliers.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig5_sensitivity_outliers")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running main simulation (R=200)...")
    ameans, bmeans = run_simulation(R=200)

    print("\nGenerating figures...")
    fig_scatter(ameans, bmeans)
    fig_outlier_boxplot(ameans, bmeans)
    fig_variance_comparison(ameans, bmeans)
    fig_single_draw_example()

    print("\nRunning sensitivity analysis (R=200, varying outlier count)...")
    fig_sensitivity_n_outliers(R=200)

    print(f"\nAll figures saved to {OUTPUT_DIR}")

    # Print summary stats
    outlier_state = 5
    frac = np.mean(ameans[:, outlier_state] > bmeans[:, outlier_state])
    print(f"\nAM > BC on outlier state: {frac:.0%} of samples")
    print(f"AM variance on outlier state:  {ameans[:, outlier_state].var():.5f}")
    print(f"BC variance on outlier state:  {bmeans[:, outlier_state].var():.5f}")
