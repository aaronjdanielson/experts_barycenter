"""
ECB Survey of Professional Forecasters backtest.

Replication of the US SPF analysis (spf_backtest.py) on Euro-area HICP histograms.

Sample: 2010Q1 – 2024Q3 (stable 12-bin structure).
Variable: current-year annual-average HICP headline.
Realized: annual-average HICP YoY from ECB SDW (ICP.M.U2.N.000000.4.ANR).
Scoring: RPS (primary), log score (secondary).
"""
import io, os, sys, re, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).parent))
from spf_backtest import cdf_median

def _rps(pmf: np.ndarray, bin_idx: int) -> float:
    """Ranked Probability Score for any K."""
    cdf_f = np.cumsum(pmf)
    cdf_r = np.zeros(len(pmf)); cdf_r[bin_idx:] = 1.0
    return float(np.sum((cdf_f[:-1] - cdf_r[:-1]) ** 2))

def _log_score(pmf: np.ndarray, bin_idx: int, eps: float = 1e-8) -> float:
    return float(np.log(np.maximum(pmf[bin_idx], eps)))

ECB_DIR  = Path(__file__).parents[2] / "data" / "ecb_spf"
OUT_DIR  = Path(__file__).parents[2] / "output" / "ecb_spf_backtest"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Bin structure (12 bins, 0.5 pp wide, 2010Q1-2024Q3) ─────────────────────
# Upper boundaries: bin k covers (-∞, -1.0], (-1.0,-0.5], ..., (3.5,4.0], [4.0,∞)
BIN_UPPERS_ECB = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, float("inf")]
K = len(BIN_UPPERS_ECB)  # 12

# Column names for the 12-bin standard format
COLS_12 = [
    "TN1_0", "FN1_0TN0_6", "FN0_5TN0_1",
    "F0_0T0_4", "F0_5T0_9",
    "F1_0T1_4", "F1_5T1_9",
    "F2_0T2_4", "F2_5T2_9",
    "F3_0T3_4", "F3_5T3_9",
    "F4_0",
]
# Extended format (2022Q3-2024Q3): last 3 collapse into F4_0
COLS_14_TAIL = ["F4_0T4_4", "F4_5T4_9", "F5_0"]


def realized_bin_ecb(value: float) -> int:
    """Map annual-average HICP YoY to a bin index 0..11."""
    for i, upper in enumerate(BIN_UPPERS_ECB):
        if value < upper:
            return i
    return K - 1


def fetch_hicp_annual() -> pd.Series:
    """
    Download monthly HICP YoY from ECB SDW and return annual averages
    indexed by year (int).
    """
    url = ("https://data-api.ecb.europa.eu/service/data/"
           "ICP/M.U2.N.000000.4.ANR"
           "?format=csvdata&detail=dataonly&startPeriod=2010-01")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    # Column 'TIME_PERIOD' has format 'YYYY-MM'
    df["year"]  = df["TIME_PERIOD"].str[:4].astype(int)
    df["value"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    annual = df.groupby("year")["value"].mean()
    return annual


def parse_hicp_block(filepath: Path, survey_year: int) -> np.ndarray | None:
    """
    Parse the headline HICP section of an ECB SPF CSV file.
    Returns (N, 12) array of valid probability vectors for current-year forecasts,
    or None if the file cannot be used.
    """
    with open(filepath, "r") as fh:
        raw = fh.read()

    lines = raw.splitlines()
    if len(lines) < 3:
        return None

    # Find the end of the HICP section (first non-HICP section header after data)
    end = len(lines)
    for i, line in enumerate(lines[2:], start=2):
        if line and not line[0].isdigit() and not line[0] == '.':
            # Section boundary: a line that's a label, not a data row
            if i > 2:
                end = i
                break

    # Grab header (line 1) and data block
    header_line = lines[1]
    data_block   = "\n".join(lines[2:end])

    cols = [c.strip() for c in header_line.split(",")]
    # Detect format
    has_14 = "F4_0T4_4" in cols

    try:
        df = pd.read_csv(io.StringIO(header_line + "\n" + data_block),
                         header=0, dtype=str)
    except Exception:
        return None

    # Current-year rows only
    df = df[df["TARGET_PERIOD"].astype(str).str.strip() == str(survey_year)].copy()
    if df.empty:
        return None

    # Extract bin columns
    if has_14:
        present = [c for c in COLS_12[:-1] if c in df.columns]  # first 11
        tail_cols = [c for c in COLS_14_TAIL if c in df.columns]
        if len(present) < 11 or not tail_cols:
            return None
        df["_F4_0_combined"] = df[tail_cols].apply(
            pd.to_numeric, errors="coerce").sum(axis=1)
        bin_data = pd.concat(
            [df[present].apply(pd.to_numeric, errors="coerce"),
             df[["_F4_0_combined"]]], axis=1
        ).values
    else:
        present = [c for c in COLS_12 if c in df.columns]
        if len(present) < 12:
            return None
        bin_data = df[present].apply(pd.to_numeric, errors="coerce").values

    # Filter: keep rows that sum to ~100 and have no NaN
    pmfs = []
    for row in bin_data:
        row = np.where(np.isnan(row), 0.0, row)
        total = row.sum()
        if abs(total - 100.0) > 5.0 or total <= 0:
            continue
        pmfs.append(row / total)

    return np.array(pmfs) if len(pmfs) >= 5 else None


def load_all_ecb_quarters(start_year: int = 2010, end_qtr: str = "2024Q3") -> dict:
    """Return {(year, qtr): (N, K) ndarray} for all usable surveys."""
    ey, eq = int(end_qtr[:4]), int(end_qtr[5])
    panel = {}
    for f in sorted(ECB_DIR.glob("*.csv")):
        m = re.match(r"(\d{4})Q([1-4])\.csv", f.name)
        if not m:
            continue
        yr, qt = int(m.group(1)), int(m.group(2))
        if yr < start_year:
            continue
        if (yr, qt) > (ey, eq):
            continue
        pmfs = parse_hicp_block(f, yr)
        if pmfs is not None:
            panel[(yr, qt)] = pmfs
    return panel


# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Fetching realized HICP …")
    hicp_annual = fetch_hicp_annual()
    print(hicp_annual.tail())

    print("\nLoading ECB SPF panel …")
    panel = load_all_ecb_quarters()
    print(f"  {len(panel)} usable quarter-surveys")

    records = []
    for (yr, qt), pmfs in sorted(panel.items()):
        if yr not in hicp_annual.index:
            continue
        realized = hicp_annual.loc[yr]
        if np.isnan(realized):
            continue
        bin_idx = realized_bin_ecb(realized)
        am  = pmfs.mean(axis=0)
        bc  = cdf_median(pmfs)
        disp = float(np.var(pmfs @ np.arange(K)))
        t    = yr + (qt - 1) / 4
        records.append(dict(
            t=t, year=yr, qtr=qt,
            realized=realized, bin_idx=bin_idx,
            n_forecasters=len(pmfs),
            am_rps=_rps(am, bin_idx),
            bc_rps=_rps(bc, bin_idx),
            am_log=_log_score(am, bin_idx),
            bc_log=_log_score(bc, bin_idx),
            am_mass=am[bin_idx],
            bc_mass=bc[bin_idx],
            dispersion=disp,
        ))

    df = pd.DataFrame(records).sort_values("t").reset_index(drop=True)
    df["hi_disp"] = df["dispersion"] >= df["dispersion"].median()
    df.to_csv(OUT_DIR / "ecb_backtest.csv", index=False)

    N = len(df)
    print(f"\n{'='*55}")
    print(f"ECB SPF backtest  ({df['year'].min()}Q{df['qtr'].min()}"
          f" – {df['year'].max()}Q{df['qtr'].max()}, n={N})")
    print(f"{'='*55}")
    print(f"\nPanel A: Overall")
    print(f"  AM  mean RPS  : {df['am_rps'].mean():.4f}")
    print(f"  BC  mean RPS  : {df['bc_rps'].mean():.4f}")
    diff_rps = df["am_rps"] - df["bc_rps"]
    stat, p = wilcoxon(diff_rps)
    print(f"  BC wins RPS   : {(diff_rps>0).mean()*100:.1f}%  Wilcoxon p={p:.4f}")

    print(f"\n  AM  mean log  : {df['am_log'].mean():.4f}")
    print(f"  BC  mean log  : {df['bc_log'].mean():.4f}")
    diff_log = df["bc_log"] - df["am_log"]  # positive = BC wins
    stat, p = wilcoxon(diff_log)
    print(f"  BC wins log   : {(diff_log>0).mean()*100:.1f}%  Wilcoxon p={p:.4f}")

    print(f"\n  AM  mass correct: {df['am_mass'].mean():.4f}")
    print(f"  BC  mass correct: {df['bc_mass'].mean():.4f}")
    diff_mass = df["bc_mass"] - df["am_mass"]
    stat, p = wilcoxon(diff_mass)
    print(f"  BC wins mass  : {(diff_mass>0).mean()*100:.1f}%  Wilcoxon p={p:.4f}")

    print(f"\nPanel B: By dispersion")
    for label, sub in [("Low", df[~df["hi_disp"]]), ("High", df[df["hi_disp"]])]:
        d = sub["am_rps"] - sub["bc_rps"]
        print(f"  {label} (n={len(sub)}): AM={sub['am_rps'].mean():.4f} "
              f"BC={sub['bc_rps'].mean():.4f}  Δ={d.mean():+.4f}")

    print(f"\nSaved to {OUT_DIR}/ecb_backtest.csv")
