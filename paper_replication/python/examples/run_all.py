"""
run_all.py — full replication pipeline for
"Robust Aggregation of Expert Probability Forecasts via Wasserstein Barycenters"

Run from repository root:
    python python/examples/run_all.py [--skip-mmlu] [--figs-only]

Options:
    --skip-mmlu   Skip the MMLU LLM ensemble step (requires API keys)
    --figs-only   Only regenerate figures from existing output CSVs
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
PYTHON = sys.executable


def run(script: str, label: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    result = subprocess.run(
        [PYTHON, str(ROOT / "python" / "examples" / script)],
        cwd=ROOT,
    )
    if result.returncode != 0:
        print(f"\n[ERROR] {script} failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-mmlu", action="store_true",
                        help="Skip MMLU step (requires OpenAI/Groq API keys)")
    parser.add_argument("--figs-only", action="store_true",
                        help="Only regenerate figures from existing CSVs")
    args = parser.parse_args()

    if not args.figs_only:
        # ── Backtests ─────────────────────────────────────────────────────────
        run("spf_backtest.py",
            "Step 1/8 — US SPF historical backtest (2007–2025)")

        run("spf_blend.py",
            "Step 2/8 — Adaptive blend OOS (2008–2025)")

        run("gjp_backtest.py",
            "Step 3/8 — GJP binary questions (Year 1, 94 questions)")

        run("gjp_multicategory_backtest.py",
            "Step 4/8 — GJP multi-category (Years 1–4, 116 questions)")

        run("ecb_spf_backtest.py",
            "Step 5/8 — ECB SPF backtest")

        run("thread3_spf_experiment.py",
            "Step 6/8 — Q-Level model OOS (2017–2024)  ← main result")

        run("chaosnli_analysis.py",
            "Step 7/8 — ChaosNLI falsification test")

        if not args.skip_mmlu:
            run("thread3_mmlu.py",
                "Step 8/8 — MMLU LLM ensemble (requires API keys)")
        else:
            print("\n[SKIPPED] Step 8/8 — MMLU LLM ensemble (--skip-mmlu)")

    # ── Figures ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Generating figures")
    print(f"{'='*60}")

    run("spf_backtest_figures.py",        "Fig: SPF timeseries and score differences")
    run("spf_backtest_blend_figure.py",  "Fig: SPF blend / mass-on-correct-bin")
    run("fig_regime_plot.py",            "Fig: ΔRPS vs dispersion (regime plot)")
    run("fig_leaderboard.py",            "Fig: OOS performance leaderboard")
    run("fig_qlevel_curve.py",           "Fig: Learned q_θ(D) curve")
    run("fig_geometry.py",               "Fig: CDF fan chart + bootstrap bands")
    run("ecb_spf_figures.py",            "Fig: ECB mass-on-correct-bin")
    run("gjp_figures.py",                "Fig: GJP binary Brier scores")
    run("gjp_multicategory_figures.py",  "Fig: GJP multi-category results")

    print(f"\n{'='*60}")
    print("  Replication complete.  Figures written to figures/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
