"""
Good Judgment Project — binary questions backtest.

For each closed binary question in GJP Year 1 with N≥20 forecasters,
computes:
  - AM   : arithmetic mean of individual probabilities of option 'a'
  - BC   : CDF-median (= median for K=2)
  - Brier scores for AM and BC

Uses the LAST forecast submitted by each user before the question closed.

Output: output/gjp/gjp_backtest.csv
        columns: n, realized, p_am, p_bc, dispersion, bs_am, bs_bc

Run from repository root:
    python python/examples/gjp_backtest.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "data" / "gjp"
OUT_DIR  = ROOT / "output" / "gjp"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_FORECASTERS = 20


# ── data loading ──────────────────────────────────────────────────────────────

def load_ifps() -> pd.DataFrame:
    """Load question metadata including outcome and number of options."""
    return pd.read_csv(
        DATA_DIR / "ifps.csv",
        lineterminator="\r",
        encoding="latin-1",
        on_bad_lines="skip",
    )


def load_survey_yr1() -> pd.DataFrame:
    """Load Year-1 individual forecasts (closed questions only)."""
    df = pd.read_csv(DATA_DIR / "survey_fcasts.yr1.tab", sep="\t")
    return df[df["q_status"] == "closed"].copy()


# ── aggregators ───────────────────────────────────────────────────────────────

def brier_score(p: float, realized: float) -> float:
    """Binary Brier score: (p - y)^2."""
    return (p - realized) ** 2


# ── main backtest ─────────────────────────────────────────────────────────────

def run_binary_backtest() -> pd.DataFrame:
    ifps = load_ifps()
    binary_ids = set(ifps.loc[ifps["n_opts"] == 2, "ifp_id"])
    outcome_map = dict(zip(ifps["ifp_id"], ifps["outcome"]))

    forecasts = load_survey_yr1()

    # Keep option-'a' rows only (for binary, value = p(option a))
    fa = forecasts[forecasts["answer_option"] == "a"].copy()
    fa["fcast_date"] = pd.to_datetime(fa["fcast_date"])

    # Last forecast per (user, question)
    last = (
        fa.sort_values("fcast_date")
        .groupby(["ifp_id", "user_id"])["value"]
        .last()
        .reset_index()
    )
    last.columns = ["ifp_id", "user_id", "p_a"]

    rows = []
    for qid, group in last.groupby("ifp_id"):
        if qid not in binary_ids:
            continue
        if len(group) < MIN_FORECASTERS:
            continue
        outcome = outcome_map.get(qid)
        if pd.isna(outcome):
            continue

        p_vals = group["p_a"].values.astype(float)
        realized = 1.0 if outcome == "a" else 0.0

        p_am = float(np.mean(p_vals))
        p_bc = float(np.median(p_vals))      # CDF-median = median for K=2
        disp = float(np.std(p_vals, ddof=1))

        rows.append({
            "n":          len(p_vals),
            "realized":   realized,
            "p_am":       p_am,
            "p_bc":       p_bc,
            "dispersion": disp,
            "bs_am":      brier_score(p_am, realized),
            "bs_bc":      brier_score(p_bc, realized),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "gjp_backtest.csv", index=False)
    print(f"Wrote {len(df)} binary questions → output/gjp/gjp_backtest.csv")
    print(f"  Mean BS: AM={df['bs_am'].mean():.4f}  BC={df['bs_bc'].mean():.4f}")
    print(f"  BC wins: {(df['bs_bc'] < df['bs_am']).mean()*100:.1f}%")
    return df


if __name__ == "__main__":
    run_binary_backtest()
