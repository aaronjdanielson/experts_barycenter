"""
ChaosNLI Analysis: BC vs. AM vs. TM on Crowdsourced NLI Annotations
=====================================================================

ChaosNLI provides 100 annotator votes per example for 1,514 NLI examples
drawn from SNLI, MNLI, and αNLI.  Each annotator assigns one label from
{Entailment, Neutral, Contradiction} (K=3, genuinely unordered).

We treat each annotator's label as a degenerate probability vector (one-hot),
aggregate across the 100 annotators, and compare BC, AM, and TM aggregates
against the original (majority) gold label.

Key prediction from Proposition C1 (K=3, (K-2)/K = 1/3):
  - BC should beat AM when annotator biases are directional and homogeneous
    within annotator subgroups (hedgers → Neutral, skeptics → Contradiction)
  - TM's rank ordering (E=1, N=2, C=3) is arbitrary; a relabeling E↔C
    reverses which extreme is trimmed — BC is invariant to this
  - Low-dispersion panels (most annotators agree) should show larger BC advantage

Usage:
    cd /Users/aarondanielson/Dropbox/experts_barycenter
    /Users/aarondanielson/miniforge3/envs/supercluster/bin/python \\
        python/examples/chaosnli_analysis.py

Interpreter: /Users/aarondanielson/miniforge3/envs/supercluster/bin/python3.11
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from wbarycenter.aggregators import (
    bc_l1, am as am_agg, tm, perm_am,
    brier_score, panel_dispersion_fast, test_equivariance,
)

ROOT    = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "output" / "chaosnli"
FIG_DIR = ROOT.parent / "figures"  # repo root figures/
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

LABEL_ORDER = ["E", "N", "C"]   # Entailment=0, Neutral=1, Contradiction=2
K = 3

plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 10, "axes.titlesize": 11,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load ChaosNLI
# ─────────────────────────────────────────────────────────────────────────────

def load_chaosnli():
    """
    Load ChaosNLI via HuggingFace datasets.
    Falls back to local JSON if dataset download fails.
    Returns DataFrame with columns:
        uid, source (snli/mnli/ab), gold_label (0/1/2),
        votes (K=3 array of vote counts, sums to 100)
    """
    try:
        from datasets import load_dataset
        print("Loading ChaosNLI from HuggingFace...")
        ds = load_dataset("copenlu/chaosnli")
        # Each split is a separate source (snli/mnli/ab)
        records = []
        for split_name, split_data in ds.items():
            for row in split_data:
                label_counts = row.get("label_counter", {})
                votes = np.array([
                    label_counts.get("e", label_counts.get("entailment", 0)),
                    label_counts.get("n", label_counts.get("neutral", 0)),
                    label_counts.get("c", label_counts.get("contradiction", 0)),
                ], dtype=float)
                total = votes.sum()
                if total < 10:
                    continue
                # Gold label: majority vote among annotators
                gold = int(np.argmax(votes))
                records.append({
                    "uid": row.get("uid", f"{split_name}_{len(records)}"),
                    "source": split_name,
                    "gold_label": gold,
                    "votes": votes,
                    "n_annotators": int(total),
                })
        print(f"  Loaded {len(records)} examples from HuggingFace")
        return pd.DataFrame(records)
    except Exception as e:
        print(f"  HuggingFace load failed ({e}); trying local files...")
        return load_chaosnli_local()


def load_chaosnli_local():
    """Load from local JSONL files if present.

    Expects data at data/chaosnli/chaosNLI_v1.0/*.jsonl
    (downloaded from https://www.dropbox.com/s/h4j7dqszmpt2679/chaosNLI_v1.0.zip).

    label_count field: [count_E, count_N, count_C], sums to 100.
    old_label field: original SNLI/MNLI gold label ('e', 'n', 'c').
    Only SNLI and MNLI files are loaded (K=3 NLI); alphanli (K=2) is skipped.
    """
    OLD_LABEL_MAP = {"e": 0, "n": 1, "c": 2}
    data_dir = ROOT / "data" / "chaosnli" / "chaosNLI_v1.0"
    records = []
    for fpath in sorted(data_dir.glob("*.jsonl")):
        if "alphanli" in fpath.stem:
            continue   # abductive NLI is binary (K=2); skip
        source = "snli" if "snli" in fpath.stem else "mnli"
        with open(fpath) as f:
            for line in f:
                row = json.loads(line.strip())
                votes = np.array(row.get("label_count", [0, 0, 0]), dtype=float)
                if votes.sum() < 10:
                    continue
                old_lbl = row.get("old_label", "")
                gold = OLD_LABEL_MAP.get(old_lbl, int(np.argmax(votes)))
                records.append({
                    "uid": row.get("uid", str(len(records))),
                    "source": source,
                    "gold_label": gold,
                    "votes": votes,
                    "n_annotators": int(votes.sum()),
                })
    if not records:
        raise FileNotFoundError(
            "No ChaosNLI data found. Download from:\n"
            "  curl -L -o data/chaosnli/chaosNLI_v1.0.zip "
            "https://www.dropbox.com/s/h4j7dqszmpt2679/chaosNLI_v1.0.zip\n"
            "  cd data/chaosnli && unzip chaosNLI_v1.0.zip"
        )
    print(f"  Loaded {len(records)} examples from local files "
          f"({sum(1 for r in records if r['source']=='snli')} SNLI, "
          f"{sum(1 for r in records if r['source']=='mnli')} MNLI)")
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Per-example aggregation
# ─────────────────────────────────────────────────────────────────────────────

def annotators_to_panel(votes: np.ndarray) -> np.ndarray:
    """
    Convert vote counts to a panel of annotator probability vectors.

    Each annotator contributes a one-hot vector; we expand the vote counts
    back to an (n_annotators, K) panel of one-hot rows.
    For large n, we instead return the normalized vote distribution as a
    single row (deterministic case) — but we need the panel to compute BC.

    For efficiency: if all annotators gave identical labels, the panel is
    just votes[k] copies of e_k.  We return the (n, K) panel directly.
    """
    n = int(votes.sum())
    K = len(votes)
    panel = np.zeros((n, K))
    row = 0
    for k, count in enumerate(votes):
        for _ in range(int(count)):
            panel[row, k] = 1.0
            row += 1
    return panel


def compute_aggregates(row: pd.Series) -> dict:
    """Compute BC, AM, TM, perm_AM, and dispersion for one example."""
    votes = row["votes"]
    panel = annotators_to_panel(votes)          # (n, K) one-hot rows
    n = panel.shape[0]

    bc  = bc_l1(panel)
    a   = am_agg(panel)
    t   = tm(panel, alpha=0.10)
    pa  = perm_am(panel)
    disp = panel_dispersion_fast(panel)

    gold = int(row["gold_label"])

    return {
        "bs_bc":    brier_score(bc,  gold),
        "bs_am":    brier_score(a,   gold),
        "bs_tm":    brier_score(t,   gold),
        "bs_pam":   brier_score(pa,  gold),
        "acc_bc":   int(np.argmax(bc)  == gold),
        "acc_am":   int(np.argmax(a)   == gold),
        "acc_tm":   int(np.argmax(t)   == gold),
        "acc_pam":  int(np.argmax(pa)  == gold),
        "dispersion": disp,
        "n_annotators": n,
        "majority_frac": float(votes.max() / votes.sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Group-level aggregation (probability distribution inputs)
# ─────────────────────────────────────────────────────────────────────────────

def compute_group_aggregates(row: pd.Series, n_group: int = 10,
                              n_reps: int = 20, seed: int = 0) -> dict:
    """
    Group-level analysis: randomly split the 100 annotators into n_groups = n/n_group
    groups; each group provides a smooth probability distribution (vote counts / n_group).
    Aggregate the resulting distributions using BC, AM, TM.

    This is the correct analog to the paper's SPF setting, where each "forecaster"
    provides a full probability distribution rather than a point prediction.

    n_reps bootstrap repetitions are used to reduce variance from random grouping.
    """
    votes = row["votes"]
    gold  = int(row["gold_label"])
    n     = int(votes.sum())
    K     = len(votes)
    n_groups = n // n_group
    if n_groups < 3:
        return None

    rng = np.random.default_rng(seed)
    bs_bc_reps, bs_am_reps, bs_tm_reps = [], [], []

    for _ in range(n_reps):
        # Reconstruct annotator labels from counts
        labels = np.repeat(np.arange(K), votes.astype(int))
        rng.shuffle(labels)
        # Split into groups
        groups = np.array_split(labels[:n_groups * n_group], n_groups)
        # Each group → distribution
        panel = np.zeros((n_groups, K))
        for g, grp in enumerate(groups):
            for lbl in grp:
                panel[g, int(lbl)] += 1
            panel[g] /= panel[g].sum()

        bc_v = bc_l1(panel)
        am_v = am_agg(panel)
        tm_v = tm(panel, alpha=0.10)
        bs_bc_reps.append(brier_score(bc_v, gold))
        bs_am_reps.append(brier_score(am_v, gold))
        bs_tm_reps.append(brier_score(tm_v, gold))

    return {
        "bs_bc_grp": float(np.mean(bs_bc_reps)),
        "bs_am_grp": float(np.mean(bs_am_reps)),
        "bs_tm_grp": float(np.mean(bs_tm_reps)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Equivariance demonstration
# ─────────────────────────────────────────────────────────────────────────────

def run_equivariance_test(df: pd.DataFrame, n_trials: int = 100) -> dict:
    """
    Test BC equivariance and TM non-equivariance on ChaosNLI panels.

    For K=3 unordered labels, interior transpositions are (E↔N) and (N↔C).
    Boundary transposition (E↔C) always leaves TM equivariant by symmetry.
    """
    rng = np.random.default_rng(0)
    sample = df.sample(min(n_trials, len(df)), random_state=0)

    results = {"bc_eq_EN": [], "tm_eq_EN": [], "bc_eq_NC": [], "tm_eq_NC": []}

    for _, row in sample.iterrows():
        panel = annotators_to_panel(row["votes"])
        if panel.shape[0] < 5:
            continue

        # Interior transposition E↔N (perm [1,0,2])
        r_en = test_equivariance(panel, [1, 0, 2])
        results["bc_eq_EN"].append(r_en["bc_equivariant"])
        results["tm_eq_EN"].append(r_en["tm_equivariant"])

        # Interior transposition N↔C (perm [0,2,1])
        r_nc = test_equivariance(panel, [0, 2, 1])
        results["bc_eq_NC"].append(r_nc["bc_equivariant"])
        results["tm_eq_NC"].append(r_nc["tm_equivariant"])

    return {k: np.mean(v) for k, v in results.items() if v}


# ─────────────────────────────────────────────────────────────────────────────
# 4. Figures
# ─────────────────────────────────────────────────────────────────────────────

def make_figures(results: pd.DataFrame, eq_results: dict):
    """
    Two-panel figure:
      Left:  equivariance rates — BC vs TM under E↔N and N↔C transpositions
      Right: group-level mean Brier by dispersion quartile (distributional inputs)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # ── Left: equivariance bar chart ─────────────────────────────────────────
    labels_eq = ["E↔N", "N↔C"]
    bc_rates = [eq_results["bc_eq_EN"] * 100, eq_results["bc_eq_NC"] * 100]
    tm_rates = [eq_results["tm_eq_EN"] * 100, eq_results["tm_eq_NC"] * 100]
    x_eq = np.arange(len(labels_eq))
    w = 0.32
    ax1.bar(x_eq - w/2, bc_rates, w, color="#1f77b4", alpha=0.9, label="BC ($\\ell^1$)")
    ax1.bar(x_eq + w/2, tm_rates, w, color="#ff7f0e", alpha=0.9, label="TM (10%)")
    ax1.axhline(100, ls="--", lw=0.8, color="gray")
    ax1.set_xticks(x_eq)
    ax1.set_xticklabels([f"Interior transposition\n{lb}" for lb in labels_eq])
    ax1.set_ylabel("% examples with equivariant prediction")
    ax1.set_ylim(0, 115)
    ax1.set_title("Permutation equivariance\n(100 random examples)")
    ax1.legend(framealpha=0.9)
    for i, (b, t) in enumerate(zip(bc_rates, tm_rates)):
        ax1.text(i - w/2, b + 2, f"{b:.0f}%", ha="center", va="bottom", fontsize=9)
        ax1.text(i + w/2, t + 2, f"{t:.0f}%", ha="center", va="bottom", fontsize=9)

    # ── Right: group-level mean BS by dispersion quartile ────────────────────
    # Use group-level Brier if available in results; else use one-hot Brier
    has_grp = "bs_bc_grp" in results.columns
    bs_am_col = "bs_am_grp" if has_grp else "bs_am"
    bs_tm_col = "bs_tm_grp" if has_grp else "bs_tm"
    bs_bc_col = "bs_bc_grp" if has_grp else "bs_bc"
    input_note = "group-level (n=10)" if has_grp else "one-hot annotators"

    results = results.copy()
    results["disp_q"] = pd.qcut(results["dispersion"], 4,
                                labels=["Q1\n(low)", "Q2", "Q3", "Q4\n(high)"])
    grp = results.groupby("disp_q", observed=True).agg(
        am=(bs_am_col, "mean"), tm=(bs_tm_col, "mean"),
        bc=(bs_bc_col, "mean"), n=(bs_am_col, "count")
    ).reset_index()

    x = np.arange(len(grp))
    w2 = 0.25
    ax2.bar(x - w2, grp["am"], w2, color="#d62728", alpha=0.85, label="AM")
    ax2.bar(x,      grp["tm"], w2, color="#ff7f0e", alpha=0.85, label="TM (10%)")
    ax2.bar(x + w2, grp["bc"], w2, color="#1f77b4", alpha=0.85, label="BC ($\\ell^1$)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{q}\n(n={n})" for q, n in
                         zip(grp["disp_q"], grp["n"])])
    ax2.set_ylabel("Mean Brier score")
    ax2.set_title(f"Mean Brier by panel dispersion\n({input_note})")
    ax2.legend(framealpha=0.9)

    fig.suptitle(
        "ChaosNLI (K=3 NLI): permutation equivariance and Brier score by dispersion\n"
        f"($n={len(results):,}$ examples, 100 annotators each)",
        fontsize=11
    )
    fig.tight_layout()
    for ext, kw in [(".pdf", {}), (".png", {"dpi": 200})]:
        fig.savefig(FIG_DIR / f"fig_chaosnli{ext}", bbox_inches="tight", **kw)
    plt.close(fig)
    print("Saved fig_chaosnli")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Load data
    df = load_chaosnli()
    print(f"Total examples: {len(df)}")
    print(f"Sources: {df['source'].value_counts().to_dict()}")

    # Compute per-example aggregates
    print("Computing aggregates...")
    agg_records = [compute_aggregates(row) for _, row in df.iterrows()]
    results = pd.DataFrame(agg_records)
    results["uid"]    = df["uid"].values
    results["source"] = df["source"].values

    # Save
    results.to_csv(OUT_DIR / "chaosnli_results.csv", index=False)
    print(f"Saved {len(results)} rows to output/chaosnli/chaosnli_results.csv")

    # ── Summary statistics ──────────────────────────────────────────────────
    print("\n" + "="*60)
    print("OVERALL RESULTS (K=3, n={})".format(len(results)))
    print("="*60)
    for agg, label in [("am","AM"), ("tm","TM (10%)"), ("bc","BC"), ("pam","perm-AM")]:
        bs_col, acc_col = f"bs_{agg}", f"acc_{agg}"
        print(f"  {label:<12}: Brier={results[bs_col].mean():.4f}  "
              f"Accuracy={results[acc_col].mean()*100:.1f}%")

    # Wilcoxon tests
    stat, p_bc_am  = wilcoxon(results["bs_am"],  results["bs_bc"],  alternative="greater")
    stat, p_bc_tm  = wilcoxon(results["bs_tm"],  results["bs_bc"],  alternative="greater")
    stat, p_bc_pam = wilcoxon(results["bs_pam"], results["bs_bc"], alternative="greater")
    print(f"\n  Wilcoxon p (BC < AM):     {p_bc_am:.4f}")
    print(f"  Wilcoxon p (BC < TM):     {p_bc_tm:.4f}")
    print(f"  Wilcoxon p (BC < perm-AM):{p_bc_pam:.4f}")

    # ── By dispersion ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("BY DISPERSION HALF")
    print("="*60)
    med_disp = results["dispersion"].median()
    for label, mask in [("Low  (D<median)", results["dispersion"] < med_disp),
                         ("High (D≥median)", results["dispersion"] >= med_disp)]:
        sub = results[mask]
        print(f"\n  {label} (n={len(sub)}):")
        for agg, name in [("am","AM"), ("bc","BC")]:
            print(f"    {name}: Brier={sub[f'bs_{agg}'].mean():.4f}  "
                  f"Acc={sub[f'acc_{agg}'].mean()*100:.1f}%")
        sub_stat, sub_p = wilcoxon(sub["bs_am"], sub["bs_bc"], alternative="greater")
        pct_bc = (sub["bs_bc"] < sub["bs_am"]).mean() * 100
        print(f"    BC wins {pct_bc:.0f}% of questions, p={sub_p:.4f}")

    # ── (K-2)/K factor for K=3 ─────────────────────────────────────────────
    overall_gap = results["bs_am"].mean() - results["bs_bc"].mean()
    print(f"\n  BC-AM Brier gap (one-hot inputs): {overall_gap:.5f}")
    print(f"  NOTE: For one-hot inputs, BC = majority vote, AM = frequency distribution.")
    print(f"  AM is better calibrated for one-hot inputs; BC advantage requires distributional inputs.")
    print(f"  (K-2)/K = {(K-2)/K:.3f} (predicted factor relative to K=4 gap)")

    # ── Group-level analysis (distributional inputs) ────────────────────────
    print("\n" + "="*60)
    print("GROUP-LEVEL ANALYSIS (n_group=10; distributional inputs)")
    print("="*60)
    print("  Computing group-level aggregates (n_reps=20 per example)...")
    grp_records = [compute_group_aggregates(row, n_group=10, n_reps=20)
                   for _, row in df.iterrows()]
    grp_df = pd.DataFrame([r for r in grp_records if r is not None])
    print(f"  n = {len(grp_df)} examples")
    for agg, label in [("am_grp","AM"), ("tm_grp","TM (10%)"), ("bc_grp","BC")]:
        print(f"  {label:<12}: Brier={grp_df[f'bs_{agg}'].mean():.4f}")
    grp_gap = grp_df["bs_am_grp"].mean() - grp_df["bs_bc_grp"].mean()
    _, grp_p = wilcoxon(grp_df["bs_am_grp"], grp_df["bs_bc_grp"], alternative="greater")
    _, grp_p_tm = wilcoxon(grp_df["bs_tm_grp"], grp_df["bs_bc_grp"], alternative="greater")
    print(f"\n  BC-AM gap (group-level):  {grp_gap:.5f}")
    print(f"  Wilcoxon p (BC<AM, group): {grp_p:.4f}")
    print(f"  Wilcoxon p (BC<TM, group): {grp_p_tm:.4f}")
    # By dispersion
    grp_df["dispersion"] = results["dispersion"].values[:len(grp_df)]
    med_disp_grp = grp_df["dispersion"].median()
    for lbl, mask in [("Low  (D<median)", grp_df["dispersion"] < med_disp_grp),
                       ("High (D≥median)", grp_df["dispersion"] >= med_disp_grp)]:
        sub = grp_df[mask]
        _, sp = wilcoxon(sub["bs_am_grp"], sub["bs_bc_grp"], alternative="greater")
        pct = (sub["bs_bc_grp"] < sub["bs_am_grp"]).mean() * 100
        print(f"\n  {lbl} (n={len(sub)}):")
        print(f"    AM={sub['bs_am_grp'].mean():.4f}  BC={sub['bs_bc_grp'].mean():.4f}  "
              f"BC wins {pct:.0f}%  p={sp:.4f}")

    # ── Equivariance test ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("EQUIVARIANCE TEST (100 random examples)")
    print("="*60)
    eq_results = run_equivariance_test(df, n_trials=100)
    print(f"  BC equivariant under E↔N transposition: {eq_results['bc_eq_EN']*100:.0f}%")
    print(f"  TM equivariant under E↔N transposition: {eq_results['tm_eq_EN']*100:.0f}%")
    print(f"  BC equivariant under N↔C transposition: {eq_results['bc_eq_NC']*100:.0f}%")
    print(f"  TM equivariant under N↔C transposition: {eq_results['tm_eq_NC']*100:.0f}%")
    print(f"\n  => TM prediction changes in "
          f"{(1 - eq_results['tm_eq_EN'])*100:.0f}% (E↔N) and "
          f"{(1 - eq_results['tm_eq_NC'])*100:.0f}% (N↔C) of examples")
    print(f"     under arbitrary relabeling of unordered categories.")

    # Merge group-level results into results df for figure
    if len(grp_df) == len(results):
        results = results.copy()
        results["bs_bc_grp"] = grp_df["bs_bc_grp"].values
        results["bs_am_grp"] = grp_df["bs_am_grp"].values
        results["bs_tm_grp"] = grp_df["bs_tm_grp"].values

    # ── Figures ─────────────────────────────────────────────────────────────
    make_figures(results, eq_results)

    print("\nDone.")
    return results


if __name__ == "__main__":
    main()
