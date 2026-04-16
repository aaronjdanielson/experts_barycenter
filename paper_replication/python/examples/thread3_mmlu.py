"""
Thread 3: LLM Ensemble Example — MMLU Position Bias and Aggregation
====================================================================

Demonstrates the aggregation framework on an LLM ensemble:
  - Three LLMs (gpt-4o-mini, gpt-3.5-turbo, qwen3-32b) queried on 100 MMLU questions
  - Four cyclic permutations of answer options per model
  - Key findings:
    1. Clear position bias: p(correct|pos A) >> p(correct|pos D)
    2. Averaging over permutations eliminates position bias (permutation invariance)
    3. With n=3 models, BC can differ from AM; coordinatewise median protects against
       weaker model contaminating the ensemble
    4. Qwen3 (96% accuracy) is a dominant model; BC protects ensemble from gpt-3.5 drag

Data: output/mmlu/mmlu_probs.parquet (created by mmlu_inference.py + mmlu_qwen3_inference.py)

Interpreter: /Users/aarondanielson/miniforge3/envs/supercluster/bin/python3.11
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import wilcoxon

ROOT = Path(__file__).parent.parent.parent
OUT_DIR = ROOT / "output" / "thread3"
FIG_DIR = ROOT.parent / "figures"  # repo root figures/
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = ROOT / "output" / "mmlu" / "mmlu_probs.parquet"
PANEL_COLS = ["p_0", "p_1", "p_2", "p_3"]
LETTER_COLS = ["p_A", "p_B", "p_C", "p_D"]
WORKING_MODELS = ["gpt4o_mini", "gpt35_turbo", "qwen3_32b_groq"]
MODEL_LABELS   = {"gpt4o_mini": "GPT-4o-mini", "gpt35_turbo": "GPT-3.5-turbo", "qwen3_32b_groq": "Qwen3-32B"}


def brier_score(pred: np.ndarray, answer_idx: int) -> float:
    one_hot = np.zeros(len(pred))
    one_hot[answer_idx] = 1.0
    return float(np.sum((pred - one_hot) ** 2))


def main():
    print("Loading MMLU data...")
    df_full = pd.read_parquet(DATA_FILE)

    # Keep only models with valid logprob-derived distributions
    df = df_full[df_full["model"].isin(WORKING_MODELS)].copy()
    print(f"  {len(df)} rows | {df.question_id.nunique()} questions | {df.model.nunique()} models | "
          f"{df.perm_idx.nunique()} permutations")
    print(f"  Models: {list(df['model'].unique())}")
    print(f"  Subjects: {df.subject.nunique()}")

    # ── Position bias analysis ─────────────────────────────────────────────────
    print("\n=== Position Bias Analysis ===")
    print("(p(correct) as function of answer position in prompt)")

    pos_records = []
    for _, row in df.iterrows():
        perm = list(row["perm"])
        correct_orig = int(row["answer_idx"])
        correct_pos = perm.index(correct_orig)
        p_correct_in_pos = [row["p_A"], row["p_B"], row["p_C"], row["p_D"]][correct_pos]
        pos_records.append({
            "model": row["model"],
            "perm_idx": row["perm_idx"],
            "correct_pos": correct_pos,
            "p_correct": p_correct_in_pos,
        })
    pos_df = pd.DataFrame(pos_records)

    labels = ["A", "B", "C", "D"]
    print("\nOverall p(correct) by position (pooled, both models):")
    overall_by_pos = []
    for pos in range(4):
        sub = pos_df[pos_df["correct_pos"] == pos]
        m = sub["p_correct"].mean()
        overall_by_pos.append(m)
        print(f"  Position {labels[pos]}: n={len(sub):4d}, mean p(correct)={m:.4f}")

    print("\nPosition bias by model:")
    model_pos = {}
    for model in WORKING_MODELS:
        sub_m = pos_df[pos_df["model"] == model]
        row_vals = []
        for pos in range(4):
            sub_p = sub_m[sub_m["correct_pos"] == pos]
            row_vals.append(sub_p["p_correct"].mean())
        model_pos[model] = row_vals
        print(f"  {model:<25}: {' '.join(f'{v:.3f}' for v in row_vals)}  "
              f"(A-D gap: {row_vals[0]-row_vals[3]:.3f})")

    # ── Permutation averaging effect ──────────────────────────────────────────
    print("\n=== Effect of Permutation Averaging ===")
    # For each (model, question): compare p(correct) in natural ordering vs perm-averaged
    nat_results, avg_results = [], []
    for model in WORKING_MODELS:
        sub_m = df[df["model"] == model]
        for qid, grp in sub_m.groupby("question_id"):
            ans = int(grp["answer_idx"].iloc[0])
            # Natural permutation (perm_idx=0)
            nat = grp[grp["perm_idx"] == 0][PANEL_COLS].values[0]
            # Perm-averaged (canonical-space average across 4 perms)
            avg = grp[PANEL_COLS].values.mean(axis=0)
            nat_results.append(brier_score(nat, ans))
            avg_results.append(brier_score(avg, ans))

    nat_arr = np.array(nat_results)
    avg_arr = np.array(avg_results)
    _, p_perm = wilcoxon(avg_arr, nat_arr)
    print(f"Natural-order BS:   {nat_arr.mean():.4f}")
    print(f"Perm-averaged BS:   {avg_arr.mean():.4f}  ({(avg_arr.mean()/nat_arr.mean()-1)*100:+.1f}%, p={p_perm:.4f})")

    # ── Ensemble aggregation (3-model panel) ────────────────────────────────────
    print("\n=== Ensemble Aggregation (perm-averaged models, n=3) ===")
    ens_results = []
    for qid, grp in df.groupby("question_id"):
        ans = int(grp["answer_idx"].iloc[0])
        model_avgs = {}
        for model in WORKING_MODELS:
            sub = grp[grp["model"] == model][PANEL_COLS].values
            if len(sub) > 0:
                model_avgs[model] = sub.mean(axis=0)
        if len(model_avgs) < 2:
            continue
        stack = np.array(list(model_avgs.values()))  # (n_models, 4)
        am = stack.mean(axis=0)
        bc = np.median(stack, axis=0); bc /= bc.sum()
        bs_per = {m: brier_score(v, ans) for m, v in model_avgs.items()}
        rec = {
            "qid": qid,
            "bs_am": brier_score(am, ans),
            "bs_bc": brier_score(bc, ans),
            **{f"bs_{m}": v for m, v in bs_per.items()},
        }
        # 2-model panel (gpt4o + gpt35, for comparison with prior result)
        if "gpt4o_mini" in model_avgs and "gpt35_turbo" in model_avgs:
            s2 = np.array([model_avgs["gpt4o_mini"], model_avgs["gpt35_turbo"]])
            rec["bs_am2"] = brier_score(s2.mean(axis=0), ans)
        ens_results.append(rec)
    ens_df = pd.DataFrame(ens_results)

    for model in WORKING_MODELS:
        key = f"bs_{model}"
        if key in ens_df:
            print(f"  {MODEL_LABELS[model]:<20}: BS={ens_df[key].mean():.4f}")
    print(f"  AM ensemble (n=3): BS={ens_df['bs_am'].mean():.4f}")
    print(f"  BC ensemble (n=3): BS={ens_df['bs_bc'].mean():.4f}  "
          f"({(ens_df['bs_bc'].mean()/ens_df['bs_am'].mean()-1)*100:+.1f}% vs AM)")
    if "bs_am2" in ens_df:
        print(f"  AM ensemble (n=2, gpt4o+gpt35): BS={ens_df['bs_am2'].mean():.4f}")
    _, p_bc_am = wilcoxon(ens_df['bs_bc'], ens_df['bs_am'])
    print(f"  BC vs AM Wilcoxon p={p_bc_am:.4f}")
    print(f"  BC != AM for n=3: {not np.allclose(ens_df.bs_am, ens_df.bs_bc)}")

    # ── Qwen3 position bias ───────────────────────────────────────────────────
    print("\nQwen3 position bias (for completeness):")
    if "qwen3_32b_groq" in pos_df["model"].values:
        sub_q = pos_df[pos_df["model"] == "qwen3_32b_groq"]
        for pos in range(4):
            sub_p = sub_q[sub_q["correct_pos"] == pos]
            if len(sub_p) > 0:
                print(f"  Position {'ABCD'[pos]}: n={len(sub_p)}, mean p={sub_p.p_correct.mean():.4f}")

    # ── Save ───────────────────────────────────────────────────────────────────
    ens_df.to_csv(OUT_DIR / "mmlu_ensemble.csv", index=False)

    # ── Figure (4 panels) ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("MMLU LLM Ensemble: Position Bias and Aggregation (3 Models)",
                 fontsize=10, fontweight="bold")

    colors_m = {"gpt4o_mini": "#2196F3", "gpt35_turbo": "#FF9800", "qwen3_32b_groq": "#4CAF50"}

    # Panel (a): Position bias by model
    ax = axes[0]
    x = np.arange(4)
    width = 0.25
    for i, model in enumerate(WORKING_MODELS):
        vals = model_pos[model]
        offset = (i - 1) * width
        ax.bar(x + offset, vals, width, label=MODEL_LABELS[model],
               color=colors_m[model], alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Answer position in prompt", fontsize=8)
    ax.set_ylabel("Mean p(correct answer)", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["A", "B", "C", "D"])
    ax.set_title("(a) Position Bias by Model", fontsize=8, fontweight="bold")
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.05)

    # Panel (b): Effect of permutation averaging (all 3 models pooled)
    ax = axes[1]
    methods = ["Natural\nordering", "Perm-averaged\n(4 perms)"]
    bs_vals = [nat_arr.mean(), avg_arr.mean()]
    bars = ax.bar(methods, bs_vals, color=["#FF9800", "#2196F3"], alpha=0.8,
                  edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, bs_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Mean Brier Score (3 models pooled)", fontsize=8)
    ax.set_title(f"(b) Permutation Averaging\n"
                 f"{(avg_arr.mean()/nat_arr.mean()-1)*100:+.1f}% (p={p_perm:.4f})",
                 fontsize=8, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ymin = min(bs_vals) * 0.95
    ymax = max(bs_vals) * 1.05
    ax.set_ylim(ymin, ymax)

    # Panel (c): Individual model BS
    ax = axes[2]
    ind_labels = [MODEL_LABELS[m].replace("-", "-\n") for m in WORKING_MODELS]
    ind_vals = [ens_df[f"bs_{m}"].mean() for m in WORKING_MODELS]
    ax.bar(ind_labels, ind_vals, color=[colors_m[m] for m in WORKING_MODELS],
           alpha=0.8, edgecolor="black", linewidth=0.5)
    for i, val in enumerate(ind_vals):
        ax.text(i, val + 0.003, f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Mean Brier Score", fontsize=8)
    ax.set_title("(c) Individual Model Performance", fontsize=8, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel (d): BC vs AM vs best individual (n=3)
    ax = axes[3]
    best_model_bs = min(ens_df[f"bs_{m}"].mean() for m in WORKING_MODELS)
    ens_labels = ["Best\nindividual", "AM\n(n=3)", "BC\n(n=3)"]
    ens_vals   = [best_model_bs, ens_df["bs_am"].mean(), ens_df["bs_bc"].mean()]
    ens_colors = ["#9E9E9E", "#FF5722", "#673AB7"]
    bars = ax.bar(ens_labels, ens_vals, color=ens_colors, alpha=0.8,
                  edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, ens_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Mean Brier Score", fontsize=8)
    ax.set_title(f"(d) Ensemble Aggregation\nBC vs AM: {(ens_df['bs_bc'].mean()/ens_df['bs_am'].mean()-1)*100:+.1f}% (p={p_bc_am:.3f})",
                 fontsize=8, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_thread3_mmlu.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure: {FIG_DIR / 'fig_thread3_mmlu.pdf'}")

    # Summary for paper
    print("\n=== Summary for paper ===")
    for m in WORKING_MODELS:
        gap = model_pos[m][0] - model_pos[m][3]
        print(f"  Position bias A−D ({MODEL_LABELS[m]}): {gap:.3f}")
    print(f"Permutation averaging: BS {nat_arr.mean():.4f} → {avg_arr.mean():.4f} "
          f"({(avg_arr.mean()/nat_arr.mean()-1)*100:+.1f}%, p={p_perm:.4f})")
    print(f"Best individual: BS={best_model_bs:.4f} ({MODEL_LABELS[min(WORKING_MODELS, key=lambda m: ens_df[f'bs_{m}'].mean())]})")
    print(f"AM ensemble (n=3): BS={ens_df['bs_am'].mean():.4f}")
    print(f"BC ensemble (n=3): BS={ens_df['bs_bc'].mean():.4f} "
          f"({(ens_df['bs_bc'].mean()/ens_df['bs_am'].mean()-1)*100:+.1f}% vs AM, p={p_bc_am:.4f})")


if __name__ == "__main__":
    main()
