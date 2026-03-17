"""
Application: Aggregating Expert Opinion on the Opioid Crisis.

Data: NYT survey of 30 public health experts allocating a hypothetical
budget across 20 interventions. Each expert's allocation is a probability
vector over the 20 interventions.

Produces all figures for Section 6 of the paper.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from wbarycenter import dw_barycenter

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "application"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "opiod"

CATEGORY_COLORS = {
    "treatment":      "#1f77b4",
    "harm reduction": "#2ca02c",
    "prevention":     "#ff7f0e",
    "supply":         "#d62728",
}


# ---------------------------------------------------------------------------
# Load and reshape data
# ---------------------------------------------------------------------------

def load_data():
    df = pd.read_csv(DATA_PATH, sep="\t")
    # pivot: rows = experts, columns = interventions
    wide = df.pivot(index="slug", columns="key", values="value")
    # keep category and intervention order
    cat = df[["key", "category"]].drop_duplicates().set_index("key")
    interventions = wide.columns.tolist()
    categories = [cat.loc[k, "category"] for k in interventions]
    experts = wide.index.tolist()
    data = wide.values.astype(float)        # (30, 20)
    return data, experts, interventions, categories


# ---------------------------------------------------------------------------
# Figure 1: Heatmap of all expert opinions
# ---------------------------------------------------------------------------

def fig_heatmap(data, experts, interventions, categories):
    n, K = data.shape
    fig, ax = plt.subplots(figsize=(14, 9))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=data.max())
    ax.set_xticks(range(K))
    ax.set_xticklabels(interventions, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(experts, fontsize=7)
    plt.colorbar(im, ax=ax, label="Budget share")
    # color x-axis labels by category
    for tick, cat in zip(ax.get_xticklabels(), categories):
        tick.set_color(CATEGORY_COLORS[cat])
    # legend for categories
    patches = [mpatches.Patch(color=c, label=cat)
               for cat, c in CATEGORY_COLORS.items()]
    ax.legend(handles=patches, loc="upper right", fontsize=8,
              title="Category", framealpha=0.9)
    ax.set_title("Budget allocations of 30 public health experts\n"
                 "across 20 opioid interventions", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_app1_heatmap.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig_app1_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_app1_heatmap")


# ---------------------------------------------------------------------------
# Figure 2: AM vs BC aggregate allocations
# ---------------------------------------------------------------------------

def fig_aggregate_comparison(am, bc, interventions, categories):
    K = len(interventions)
    x = np.arange(K)
    width = 0.38
    colors = [CATEGORY_COLORS[c] for c in categories]

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for ax, vals, title in zip(axes, [am, bc],
                                ["Arithmetic Mean", "Wasserstein Barycenter"]):
        bars = ax.bar(x, vals, color=colors, alpha=0.85, width=0.7)
        ax.set_ylabel("Budget share")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, max(am.max(), bc.max()) * 1.15)
        # annotate values > 5%
        for i, v in enumerate(vals):
            if v > 0.05:
                ax.text(i, v + 0.003, f"{v:.2f}", ha="center",
                        va="bottom", fontsize=6)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(interventions, rotation=45, ha="right", fontsize=7)
    for tick, cat in zip(axes[1].get_xticklabels(), categories):
        tick.set_color(CATEGORY_COLORS[cat])

    patches = [mpatches.Patch(color=c, label=cat)
               for cat, c in CATEGORY_COLORS.items()]
    axes[0].legend(handles=patches, loc="upper right", fontsize=8,
                   title="Category", framealpha=0.9)

    fig.suptitle("Aggregate expert opinion on opioid interventions", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_app2_aggregates.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig_app2_aggregates.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_app2_aggregates")


# ---------------------------------------------------------------------------
# Figure 3: AM vs BC side-by-side scatter per intervention
# ---------------------------------------------------------------------------

def fig_am_vs_bc_scatter(am, bc, interventions, categories):
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, (a, b, cat) in enumerate(zip(am, bc, categories)):
        ax.scatter(a, b, color=CATEGORY_COLORS[cat], s=60, zorder=3)
        if abs(a - b) > 0.02 or max(a, b) > 0.05:
            ax.annotate(interventions[i].replace("-", "\n"),
                        (a, b), fontsize=5.5, ha="left",
                        xytext=(4, 2), textcoords="offset points")
    lo, hi = 0, max(am.max(), bc.max()) * 1.1
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="AM = BC")
    patches = [mpatches.Patch(color=c, label=cat)
               for cat, c in CATEGORY_COLORS.items()]
    ax.legend(handles=patches + [plt.Line2D([0], [0], ls="--", color="k",
              label="AM = BC")], fontsize=8, framealpha=0.9)
    ax.set_xlabel("Arithmetic Mean")
    ax.set_ylabel("Wasserstein Barycenter")
    ax.set_title("AM vs. Barycenter allocation per intervention")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_app3_scatter.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig_app3_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_app3_scatter")


# ---------------------------------------------------------------------------
# Figure 4: Leave-one-out influence — how much each expert shifts AM vs BC
# ---------------------------------------------------------------------------

def fig_leave_one_out(data, experts, interventions):
    n, K = data.shape
    weights_full = np.ones(n) / n
    D = np.ones((K, K)) - np.eye(K)

    am_full = data.mean(axis=0)
    bc_full, _ = dw_barycenter(data, weights_full, D)

    am_shifts, bc_shifts = [], []
    print("  Computing leave-one-out (30 runs)...")
    for i in range(n):
        dat_loo = np.delete(data, i, axis=0)
        w_loo = np.ones(n - 1) / (n - 1)
        am_loo = dat_loo.mean(axis=0)
        bc_loo, _ = dw_barycenter(dat_loo, w_loo, D)
        am_shifts.append(np.sum(np.abs(am_loo - am_full)))
        bc_shifts.append(np.sum(np.abs(bc_loo - bc_full)))

    am_shifts = np.array(am_shifts)
    bc_shifts = np.array(bc_shifts)
    order = np.argsort(am_shifts)[::-1]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(n)
    ax.bar(x - 0.2, am_shifts[order], 0.38, label="Arithmetic Mean",
           color="#d62728", alpha=0.8)
    ax.bar(x + 0.2, bc_shifts[order], 0.38, label="Wasserstein Barycenter",
           color="#1f77b4", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([experts[i] for i in order], rotation=45,
                       ha="right", fontsize=7)
    ax.set_ylabel(r"$\ell^1$ shift in aggregate when expert removed")
    ax.set_title("Leave-one-out influence of each expert")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_app4_loo.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig_app4_loo.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_app4_loo")
    return am_shifts, bc_shifts, order, experts


# ---------------------------------------------------------------------------
# Figure 5: Category-level aggregates (collapse interventions by category)
# ---------------------------------------------------------------------------

def fig_category_aggregates(am, bc, interventions, categories):
    cats = list(CATEGORY_COLORS.keys())
    am_cat = {c: sum(am[i] for i, cat in enumerate(categories) if cat == c)
              for c in cats}
    bc_cat = {c: sum(bc[i] for i, cat in enumerate(categories) if cat == c)
              for c in cats}

    x = np.arange(len(cats))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width/2, [am_cat[c] for c in cats], width,
           label="Arithmetic Mean",
           color=[CATEGORY_COLORS[c] for c in cats], alpha=0.6,
           edgecolor="black", linewidth=0.8)
    ax.bar(x + width/2, [bc_cat[c] for c in cats], width,
           label="Wasserstein Barycenter",
           color=[CATEGORY_COLORS[c] for c in cats], alpha=1.0,
           edgecolor="black", linewidth=0.8)
    # add hatching to AM bars to distinguish
    for bar in ax.patches[:len(cats)]:
        bar.set_hatch("//")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=10)
    ax.set_ylabel("Total budget share")
    ax.set_title("Aggregate budget share by intervention category")
    hatched = mpatches.Patch(facecolor="gray", hatch="//", label="Arithmetic Mean")
    solid   = mpatches.Patch(facecolor="gray", label="Wasserstein Barycenter")
    ax.legend(handles=[hatched, solid], fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_app5_categories.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig_app5_categories.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_app5_categories")
    return am_cat, bc_cat


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading data...")
    data, experts, interventions, categories = load_data()
    print(f"  {len(experts)} experts x {len(interventions)} interventions")

    print("Computing aggregates...")
    n, K = data.shape
    weights = np.ones(n) / n
    D = np.ones((K, K)) - np.eye(K)
    am = data.mean(axis=0)
    bc, _ = dw_barycenter(data, weights, D)

    print("\n=== Top interventions by AM ===")
    order_am = np.argsort(am)[::-1]
    for i in order_am[:5]:
        print(f"  {interventions[i]:45s}  AM={am[i]:.3f}  BC={bc[i]:.3f}")

    print("\n=== Largest AM - BC differences ===")
    diff = am - bc
    for i in np.argsort(np.abs(diff))[::-1][:6]:
        print(f"  {interventions[i]:45s}  diff={diff[i]:+.3f}")

    print("\nGenerating figures...")
    fig_heatmap(data, experts, interventions, categories)
    fig_aggregate_comparison(am, bc, interventions, categories)
    fig_am_vs_bc_scatter(am, bc, interventions, categories)
    am_shifts, bc_shifts, order, experts_sorted = fig_leave_one_out(
        data, experts, interventions)
    am_cat, bc_cat = fig_category_aggregates(am, bc, interventions, categories)

    print("\n=== Category totals ===")
    for cat in CATEGORY_COLORS:
        print(f"  {cat:20s}  AM={am_cat[cat]:.3f}  BC={bc_cat[cat]:.3f}")

    print(f"\nAll figures saved to {OUTPUT_DIR}")
