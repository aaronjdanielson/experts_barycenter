# Robust Aggregation of Expert Probability Forecasts via Wasserstein Barycenters

> Danielson, A.J. and Amini, A.A. (2026). "Robust Aggregation of Expert Probability Forecasts via Wasserstein Barycenters." Submitted.

The **Wasserstein barycenter** (under the indicator ground cost, the coordinatewise CDF median) is a robust alternative to the arithmetic mean for aggregating expert probability forecasts. A learned two-parameter variant (Q-Level model) achieves −15.5% OOS RPS vs. the fixed barycenter on US SPF 2017–2024.

---

## Repository structure

```
experts_barycenter/
├── paper/
│   ├── ExpertsBarycenter.tex          # Root LaTeX file (compiles the paper)
│   ├── DiscreteBarycenter.tex         # Paper body (~1800 lines)
│   └── barybib.tex                    # Bibliography
├── figures/                           # Generated PDF figures (written by pipeline, read by paper)
├── paper_replication/                 # Self-contained replication package
│   ├── README.md                      # Full replication guide + audit log
│   ├── python/
│   │   ├── wbarycenter/               # Core library (barycenter, aggregators, learned geometry)
│   │   ├── examples/                  # Replication scripts (one per paper section)
│   │   │   └── run_all.py             # Master pipeline — runs all steps in order
│   │   ├── pyproject.toml
│   │   └── requirements.txt
│   ├── data/                          # Raw data (gitignored — see paper_replication/README.md)
│   └── output/                        # Generated CSVs (gitignored — reproduced by pipeline)
└── deprecated_markdowns/              # Legacy agent reports and working files (gitignored)
```

---

## Setup

Requires Python 3.11+. From the repository root:

```bash
pip install -r paper_replication/python/requirements.txt
pip install -e paper_replication/python/
pip install torch cvxpy   # required for Q-Level model (Step 6)
```

> **Note (Python 3.14 on macOS):** Use `python3 -m pip install ...` if `pip` points to a different Python version.

---

## Data

Data files are not tracked in git and must be downloaded separately. See **[paper_replication/README.md](paper_replication/README.md)** for download sources and exact file paths.

- **US SPF**: `paper_replication/data/spf/SPFmicrodata.xlsx` (Philadelphia Fed)
- **GJP**: `paper_replication/data/gjp/ifps.csv` + `survey_fcasts.yr1–4.tab` (Harvard Dataverse, CC0)
- **ECB SPF**: downloaded automatically via ECB SDW API
- **ChaosNLI**: `paper_replication/data/chaosnli/chaosNLI_v1.0/` (JSON files)

Set `FRED_API_KEY` for live CPI data; scripts fall back to hardcoded values (2007–2025) otherwise.

---

## Replication

From the **repository root**:

```bash
# Full pipeline (skips MMLU which requires OpenAI/Groq API keys)
python paper_replication/python/examples/run_all.py --skip-mmlu

# Figures only (from existing output CSVs)
python paper_replication/python/examples/run_all.py --figs-only
```

Steps run in order:
1. US SPF historical backtest (2007–2025, 76 quarters)
2. Adaptive blend OOS (56-quarter expanding window)
3. GJP binary backtest (Year 1, 94 questions)
4. GJP multi-category backtest (Years 1–4, 116 questions)
5. ECB SPF backtest (2010–2024, 56 quarters)
6. Q-Level model OOS (2017–2024, 32 OOS quarters) ← main result
7. ChaosNLI equivariance falsification test
8. MMLU LLM ensemble [requires API keys — use `--skip-mmlu`]

For detailed documentation including data sources, environment notes, and a full audit of replication issues encountered, see **[paper_replication/README.md](paper_replication/README.md)**.

---

## Key results

| Experiment | Result |
|---|---|
| US SPF (2007–2025, 76 quarters): BC vs AM | 68.4% wins, Wilcoxon p=0.005 |
| GJP binary (94 questions): BC vs AM | 12.8% Brier reduction, 79/94 wins |
| GJP multi-category (116 questions): BC vs AM | 16.9% Brier reduction (p<0.001) |
| GJP multi-category (unordered, n=30): BC vs AM | 20.2% Brier reduction |
| US SPF adaptive blend (56 quarters): blend vs AM | 4.3% RPS reduction (p<0.001) |
| Q-Level model OOS (2017–2024): vs fixed BC | **−15.5% RPS** (p<0.001); −25.2% in high-dispersion quarters |
| ChaosNLI (exchangeable annotators): BC vs AM | AM wins (+8.0%), as predicted by scope conditions |

---

## License

MIT
