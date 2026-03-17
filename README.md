# Robust Aggregation of Expert Probability Forecasts via Wasserstein Barycenters

Replication code for:

> Danielson, A.J. and Amini, A.A. (2025). "Robust Aggregation of Expert Probability Forecasts via Wasserstein Barycenters." *Journal of Forecasting*.

## Repository structure

```
experts_barycenter/
├── ExpertsBarycenter.tex      # Main paper
├── DiscreteBarycenter.tex     # Included section file
├── barybib.tex                # Bibliography
├── data/
│   ├── opiod                  # NYT opioid survey data (tab-separated)
│   └── spf/                   # SPF data (see below)
├── python/
│   ├── wbarycenter/           # Python package
│   ├── examples/              # Replication scripts
│   ├── pyproject.toml
│   └── README.md              # Package documentation and API reference
└── code/                      # Legacy MATLAB code
```

## Python package

Install from GitHub:

```bash
pip install git+https://github.com/aaronjdanielson/experts_barycenter.git#subdirectory=python
```

Or clone and install in editable mode:

```bash
git clone https://github.com/aaronjdanielson/experts_barycenter
cd experts_barycenter/python
pip install -e .
```

See [python/README.md](python/README.md) for the full API and quick-start examples.

## Replication

All replication scripts are in `python/examples/` and should be run from the **repository root**:

```bash
python python/examples/robustness.py          # §3 Robustness figures
python python/examples/simulations_study.py   # §4 Simulation tables and efficiency figure
python python/examples/application.py         # §5 NYT opioid survey application
python python/examples/spf_application.py     # §6 Survey of Professional Forecasters application
```

Output figures are written to `output/`.

### SPF data

The SPF application requires microdata from the Philadelphia Fed. Download
`SPFmicrodata.xlsx` from the
[Survey of Professional Forecasters](https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/survey-of-professional-forecasters)
page and place it at `data/spf/SPFmicrodata.xlsx`.

## License

MIT
