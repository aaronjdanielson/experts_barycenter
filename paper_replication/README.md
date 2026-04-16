# Replication Package

**Paper:** "Robust Aggregation of Expert Probability Forecasts via Wasserstein Barycenters"
**Authors:** Aaron J. Danielson, Arash A. Amini

This directory contains all code required to reproduce the tables, figures, and numerical
claims in the paper. Running `python python/examples/run_all.py --skip-mmlu` from the
**repository root** regenerates every paper figure and intermediate CSV from raw data.

---

## Directory Structure

```
paper_replication/
├── python/
│   ├── wbarycenter/          Core library (barycenter, aggregators, learned geometry)
│   ├── examples/             Replication scripts (one per paper section)
│   │   ├── run_all.py        Master pipeline — runs all steps in order
│   │   ├── spf_backtest.py   US SPF historical backtest (Table 3, Figs bt1–bt8)
│   │   ├── spf_blend.py      Adaptive blend OOS evaluation (Table 4)
│   │   ├── gjp_backtest.py   GJP binary questions (Table 2, Fig gjp1)
│   │   ├── gjp_multicategory_backtest.py  GJP multi-category (Table 5, Fig gjp2)
│   │   ├── ecb_spf_backtest.py            ECB SPF backtest (Section 6.3)
│   │   ├── thread3_spf_experiment.py      Q-Level model OOS (Table 4, Figs leaderboard/geometry)
│   │   ├── chaosnli_analysis.py           ChaosNLI equivariance test (Appendix E)
│   │   ├── thread3_mmlu.py                MMLU LLM ensemble (Appendix E, requires API keys)
│   │   └── [figure scripts]   fig_*.py, *_figures.py
│   ├── pyproject.toml
│   └── requirements.txt
├── data/                     Raw data (gitignored — see Data section below)
└── output/                   Generated CSVs (gitignored — reproduced by pipeline)
```

Generated figures are written to `../figures/` (the `figures/` directory at the repository
root), which is where the LaTeX paper reads them from.

---

## Environment Setup

Requires Python 3.11+. Install dependencies:

```bash
pip install -r paper_replication/python/requirements.txt
pip install -e paper_replication/python/   # installs wbarycenter package
```

The pipeline requires `torch` and `cvxpy` for the Q-Level model (Step 6):

```bash
pip install torch cvxpy
```

**Note (Python 3.14 on macOS):** If `pip` points to a different Python version than
`python3`, use `python3 -m pip install ...` instead. See the replication audit below.

---

## Data

The following data files are not tracked in git and must be downloaded separately.

### US Survey of Professional Forecasters (Philadelphia Fed)
File: `paper_replication/data/spf/SPFmicrodata.xlsx`
Source: https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/survey-of-professional-forecasters

Download the "Probability Distribution" microdata Excel file and place it at the path above.

### Good Judgment Project (GJP) — Harvard Dataverse
Files: `paper_replication/data/gjp/`
- `ifps.csv` — question metadata
- `survey_fcasts.yr1.tab` through `survey_fcasts.yr4.tab` — forecaster-level submissions

Source: Harvard Dataverse, doi:10.7910/DVN/BPCDH5 (public, CC0)

### European Central Bank SPF
Downloaded automatically by `ecb_spf_backtest.py` via the ECB Statistical Data Warehouse API.

### ChaosNLI
File: `paper_replication/data/chaosnli/chaosNLI_v1.0/` (JSON files)
Source: https://github.com/easonnie/ChaosNLI (download chaosNLI_v1.0.zip)

### Realized CPI (FRED)
Scripts use a hardcoded fallback for US core CPI (CPILFESL, Q4/Q4, 2007–2025).
Set `FRED_API_KEY` environment variable for live FRED API access.

---

## Running the Pipeline

From the **repository root**:

```bash
# Full pipeline (skips MMLU which requires OpenAI/Groq API keys)
python paper_replication/python/examples/run_all.py --skip-mmlu

# Figures only (from existing output CSVs)
python paper_replication/python/examples/run_all.py --figs-only
```

Steps run in order:
1. US SPF backtest (2007–2025, 76 quarters)
2. Adaptive blend OOS (56-quarter expanding window)
3. GJP binary backtest (Year 1, 94 questions)
4. GJP multi-category backtest (Years 1–4, 116 questions)
5. ECB SPF backtest (2010–2024, 56 quarters)
6. Q-Level model OOS (2017–2024, 32 OOS quarters) ← main result
7. ChaosNLI equivariance falsification test
8. MMLU LLM ensemble [requires API keys — use `--skip-mmlu`]

---

## Replication Audit: Issues Encountered and Resolved

This section documents discrepancies discovered during systematic replication of every
number and figure in the paper, and how each was resolved. Numbers in the paper match
the pipeline output as of the repository's current state.

### 1. GJP Multi-Category Classifier Bug (Critical)

**Problem:** The question classifier in `gjp_multicategory_backtest.py` was applying
regex patterns to combined question-text plus options-text, not options-text only.
This caused questions like "Who will be president as of 1 January 2015?" (categorical)
to be classified as temporal (because the question text contained a date). The resulting
classification was temporal=92 / unordered=15 / ordinal=9 instead of the correct 45/30/41.

**Root cause:** Regex matched against a concatenated string; the paper methodology
specifies classification based on options text only.

**Resolution:** Rewrote `classify_question()` to use options text exclusively.
Added `_count_date_options()` requiring ≥2 options to contain date-window patterns
before classifying as temporal (prevents single-option date strings like
"(d) Not before 1 February 2013" from triggering temporal classification).
Dollar-amount pattern (`\$\s*\d+`) added to `_ORDINAL_PAT` to catch questions
like "Below $95 per barrel."

**Impact:** Corrected classification (45 temporal / 30 unordered / 41 ordinal).
All paper numbers updated accordingly (16.9% overall / 20.2% unordered vs.
the earlier erroneously reported 8.7% / 3.2%).

### 2. "Last Forecast" vs. "Mean of All Forecasts" Methodology

**Problem:** The AM Brier score for GJP multi-category came out 0.254 (code) vs.
0.332 (earlier run). The discrepancy was traced to two different methodologies:
using each forecaster's *last* submitted forecast before closure (AM=0.254, as
stated in the paper's data section) versus averaging over all submissions (AM=0.332).

**Resolution:** The paper text explicitly states "last submitted forecast before
question closure." The code implements this correctly. The discrepancy was from
an earlier analysis pass that used a different protocol. No code change required.

### 3. GJP Question Count (116 vs. 113)

**Problem:** The CLAUDE.md planning document recorded 113 multi-category questions,
but the backtest produces 116.

**Resolution:** 113 was an intermediate count from an earlier data integration pass.
All 116 questions pass the N≥20 valid-forecasters filter (minimum observed: 69).
Paper updated to 116 throughout.

### 4. SPF Quarter Count (76 vs. 72)

**Problem:** The hardcoded CPI fallback in `spf_backtest.py` and `spf_blend.py`
only included years 2007–2024, producing 72 quarters instead of the paper's stated 76.
The 2025 Q4/Q4 value was missing.

**Resolution:** Added `2025: 2.6` to the hardcoded dictionary in both scripts.
Source: BLS Consumer Price Index News Release, January 13, 2026
(12-month change in All Items Less Food and Energy, December 2025 = 2.6%).
This restored the 76-quarter count and all dependent statistics (win rate 68.4%,
Wilcoxon p=0.005, DM p=0.225) to match the paper.

### 5. `spf_blend.py` — No FRED Fallback (Pipeline Crash)

**Problem:** `spf_blend.py` called `fetch_q4q4_cpi()` at module level without
a try/except block, crashing with `RuntimeError: FRED API key not set` even though
`spf_backtest.py` had a hardcoded fallback for the same situation.

**Resolution:** Added the same try/except fallback to `spf_blend.py`.

### 6. Missing Python Dependencies (`torch`, `cvxpy`)

**Problem:** `thread3_spf_experiment.py` requires `torch` and (via the `wbarycenter`
package) `cvxpy`. These were not installed in the Python 3.14 system environment.
The system `pip` pointed to a Homebrew Python 3.11 installation, so `pip install torch`
appeared to succeed but installed into the wrong environment.

**Resolution:** Use `python3 -m pip install torch cvxpy --break-system-packages`
to install into the same Python used by `python3`.

### 7. `thread3_spf_experiment.py` — Unbounded OOS Window

**Problem:** After adding the 2025 CPI value (fix #4), the OOS window in
`thread3_spf_experiment.py` would silently extend from 32 to 36 quarters
because it filtered `year >= OOS_START` with no upper bound.

**Resolution:** Added `OOS_END_YEAR = 2024` constant and applied it in the
OOS filter. The Q-Level model is validated on the 2017–2024 holdout (T=32),
consistent with the paper text. The 2025 data extends the SPF Table 3 backtest
but not the OOS evaluation.

### 8. `spf_backtest_blend_figure.py` Missing from Pipeline

**Problem:** Figures `fig_bt6_mass_correct_bin.pdf` and `fig_bt7_blend.pdf`
are referenced in the paper but were generated by `spf_backtest_blend_figure.py`,
which was not included in `run_all.py`. Additionally, `spf_backtest_figures.py`
wrote `fig_bt1/2/3` only to `output/spf_backtest/`, not to `figures/`, leaving
stale copies in `figures/`.

**Resolution:** Added `spf_backtest_blend_figure.py` to the figure-generation
step in `run_all.py`. Updated `spf_backtest_figures.py` to write bt1/2/3/4 to
both `output/spf_backtest/` and `figures/`. Updated `spf_backtest.py` to also
write `fig_bt8_reliability_*.pdf` to `figures/`.

---

## Key Numerical Results (Current Pipeline Output)

| Claim | Value |
|-------|-------|
| GJP binary: BC vs AM Brier reduction | 12.8% (79/94 wins, p<0.001) |
| GJP multi-category: BC vs AM (overall) | 16.9% reduction (p<0.001) |
| GJP multi-category: BC vs AM (unordered, n=30) | 20.2% reduction (p<0.001) |
| US SPF: BC vs AM wins | 68.4% of 76 quarters |
| US SPF: Wilcoxon p | 0.005 (one-sided) |
| US SPF: DM p (overall) | 0.225 |
| US SPF: DM p (low-dispersion quarters) | <0.001 |
| Q-Level model vs BC (OOS 2017–2024) | −15.5% RPS (p<0.001) |
| Q-Level model vs BC (high-dispersion) | −25.2% RPS |
| ECB SPF: Wilcoxon p | 0.234 (not significant overall) |
