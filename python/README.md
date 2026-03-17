# wbarycenter

Robust aggregation of expert probability forecasts via Wasserstein barycenters.

Companion code for:

> Danielson, A.J. and Amini, A.A. (2025). "Robust Aggregation of Expert Probability Forecasts via Wasserstein Barycenters." *Journal of Forecasting*.

## Installation

```bash
pip install wbarycenter
```

Or from source:

```bash
git clone https://github.com/[add-repo-here]
cd [repo]/python
pip install -e .
```

## Quick start

### Unordered outcomes (indicator distance)

```python
import numpy as np
from wbarycenter import dw_barycenter, plot_aggregate

# Expert probability vectors: shape (n_experts, n_outcomes)
data = np.array([
    [0.1, 0.6, 0.3],
    [0.4, 0.4, 0.2],
    [0.9, 0.05, 0.05],   # outlier
    [0.2, 0.5, 0.3],
])
labels = ["Outcome A", "Outcome B", "Outcome C"]

n, K = data.shape
D = np.ones((K, K)) - np.eye(K)          # indicator distance
weights = np.ones(n) / n

bc, transport_plans = dw_barycenter(data, weights, D)
am = data.mean(axis=0)

plot_aggregate(am, bc, labels)
```

### Ordered outcomes — closed form (no solver)

When outcomes are ordered bins (e.g. histogram probability forecasts), the
W1 barycenter equals the distribution whose CDF is the component-wise median
of the individual CDFs. No LP solver needed.

```python
from wbarycenter import cdf_median, plot_cdfs

bc = cdf_median(data)          # fast closed-form result
plot_cdfs(data, am, bc, labels)
```

### Robustness diagnostics

```python
from wbarycenter import loo_influence, plot_loo

result = loo_influence(data, weights, D)
plot_loo(result, expert_labels=["Expert A", "Expert B", "Expert C", "Expert D"])

print(f"Mean AM shift:  {result['am_shifts'].mean()*100:.1f}pp")
print(f"Mean BC shift:  {result['bc_shifts'].mean()*100:.1f}pp")
```

### Scoring against a realized outcome

```python
from wbarycenter import score_summary

# outcome_bin is the 0-indexed realized bin
scores = score_summary(am, bc, outcome_bin=1, ordered=True)
print(scores)
# {'brier_am': ..., 'brier_bc': ..., 'log_score_am': ...,
#  'log_score_bc': ..., 'crps_am': ..., 'crps_bc': ...}
```

## API

| Function | Description |
|---|---|
| `dw_barycenter(data, weights, D)` | LP-based discrete Wasserstein barycenter |
| `cdf_median(data)` | Closed-form W1 barycenter for ordered outcomes |
| `plot_aggregate(am, bc, labels)` | Side-by-side bar chart |
| `plot_cdfs(data, am, bc, labels)` | CDF overlay (ordered outcomes) |
| `loo_influence(data, weights, D)` | Leave-one-out L1 shifts for each expert |
| `plot_loo(result)` | Bar chart of LOO influence |
| `crps_ordered(forecast, outcome_bin)` | CRPS for ordered categorical forecast |
| `brier_score(forecast, outcome_bin)` | Brier score |
| `score_summary(am, bc, outcome_bin)` | All scores for AM and BC in one call |

## Replication

The `examples/` directory contains scripts that reproduce all paper results:

| Script | Paper section |
|---|---|
| `robustness.py` | §3 Robustness figures |
| `simulations_study.py` | §4 Simulation tables and efficiency figure |
| `application.py` | §5 NYT opioid survey application |
| `spf_application.py` | §6 Survey of Professional Forecasters application |

Run from the repository root (not from `python/`), e.g.:

```bash
cd experts_barycenter
python python/examples/spf_application.py
```

Data for the SPF application must be downloaded separately from the
[Philadelphia Fed website](https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/survey-of-professional-forecasters).

## License

MIT
