"""
Good Judgment Project — multi-category questions backtest (Years 1–4).

For each closed question with K ≥ 3 options and N ≥ 20 forecasters
with complete option coverage, computes:
  - AM   : arithmetic mean PMF
  - BC   : coordinatewise median PMF (Theorem 2.1, indicator ground cost)
  - TM   : 10%-trimmed mean PMF
  - Brier scores for each method

Uses the LAST forecast submitted by each user before the question closed.

Question classification is based on OPTIONS TEXT only (not question text):
  "temporal"  — at least 2 options contain explicit date ranges
                (e.g. "Between 1 Jul 2012 and 30 Sep 2012")
  "ordinal"   — options contain numeric ranges/quantities with natural ordering
                (e.g. "Less than 3% / 3–4% / More than 4%")
  "unordered" — genuinely unordered categorical outcomes (names, parties,
                countries, policy outcomes without numeric ordering)

Output: output/gjp/gjp_multicategory_backtest.csv
        columns: qid, K, cat, bs_am, bs_bc, bs_tm, disp

Run from repository root:
    python python/examples/gjp_multicategory_backtest.py
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "data" / "gjp"
OUT_DIR  = ROOT / "output" / "gjp"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_FORECASTERS = 20
TRIM_FRAC       = 0.10

# ── question classification ────────────────────────────────────────────────────

_MONTH = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|"
    r"may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|"
    r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)

# Match a single option that contains an explicit date window.
# Requires either:
#   - "Between <day> <month>" / "Before <day> <month>" / "Not before <day> <month>"
#   - "On or before <day> <month>"
#   - "Yes, by <day> <month>" / "Yes, between <day> <month>"  (for conditional yes/no)
_SINGLE_OPTION_DATE = re.compile(
    r"between\s+\d{1,2}\s+" + _MONTH
    + r"|before\s+\d{1,2}\s+" + _MONTH
    + r"|not\s+before\s+\d{1,2}\s+" + _MONTH
    + r"|on\s+or\s+before\s+\d{1,2}\s+" + _MONTH
    + r"|yes,?\s+by\s+\d{1,2}\s+" + _MONTH
    + r"|yes,?\s+between\s+\d{1,2}\s+" + _MONTH,
    re.IGNORECASE,
)

_ORDINAL_PAT = re.compile(
    r"less\s+than\s+\d"
    r"|fewer\s+than\s+\d"
    r"|more\s+than\s+\d"
    r"|\d+[\.,]?\d*\s*(?:percent|%|pct)"
    r"|\$\s*\d+(?:\.\d+)?"          # dollar amounts like $95 or $1.04
    r"|\d+\s*(?:million|billion|thousand)"
    r"|\d+\s+or\s+(?:more|fewer|less)"
    r"|(?:barrels?\s+per\s+day|bpd)"
    r"|increase\s+of|decrease\s+of"
    r"|downgrade|upgrade"
    r"|not\s+in\s+any|in\s+[1-9]\s+of\s+these"
    r"|(?:^|\(.\)\s*)none,",
    re.IGNORECASE,
)


def _count_date_options(options_str: str) -> int:
    """Count individual options that contain a date-window pattern."""
    # Split on option labels (a), (b), (c), ...
    parts = re.split(r"\([a-j]\)\s*", options_str)
    return sum(1 for p in parts if _SINGLE_OPTION_DATE.search(p))


def classify_question(options_str: str) -> str:
    """
    Classify a question based on OPTIONS text only.

    Rules (applied in order):
      1. If at least 2 options contain explicit date-window text → temporal
      2. If options contain numeric-range / quantity patterns → ordinal
      3. Otherwise → unordered
    """
    if _count_date_options(options_str) >= 2:
        return "temporal"
    if _ORDINAL_PAT.search(options_str):
        return "ordinal"
    return "unordered"


# ── data loading ───────────────────────────────────────────────────────────────

def load_ifps() -> pd.DataFrame:
    return pd.read_csv(
        DATA_DIR / "ifps.csv",
        lineterminator="\r",
        encoding="latin-1",
        on_bad_lines="skip",
    )


def load_all_years() -> pd.DataFrame:
    """Load closed forecasts from Years 1–4."""
    dfs = []
    for yr in range(1, 5):
        path = DATA_DIR / f"survey_fcasts.yr{yr}.tab"
        if not path.exists():
            print(f"  Warning: {path.name} not found, skipping.")
            continue
        df = pd.read_csv(path, sep="\t")
        df = df[df["q_status"] == "closed"].copy()
        dfs.append(df)
        print(f"  Loaded yr{yr}: {len(df):,} rows")
    return pd.concat(dfs, ignore_index=True)


# ── aggregators ───────────────────────────────────────────────────────────────

def coord_median_pmf(pmf_matrix: np.ndarray) -> np.ndarray:
    """
    Coordinatewise median of PMF rows, normalized to sum to 1.
    Implements Theorem 2.1 (indicator ground cost, unordered outcome space).
    The same formula is used for all question types in the robustness panel.
    """
    med = np.median(pmf_matrix, axis=0)
    med = np.maximum(med, 0.0)
    total = med.sum()
    if total > 0:
        med /= total
    return med


def trimmed_mean_pmf(pmf_matrix: np.ndarray, trim: float = TRIM_FRAC) -> np.ndarray:
    """10%-trimmed mean, trimming on first-option probability."""
    n = pmf_matrix.shape[0]
    k = max(1, int(np.floor(trim * n)))
    order = np.argsort(pmf_matrix[:, 0])
    trimmed = pmf_matrix[order[k : n - k], :]
    if len(trimmed) == 0:
        return pmf_matrix.mean(axis=0)
    tm = trimmed.mean(axis=0)
    tm = np.maximum(tm, 0.0)
    tm /= tm.sum()
    return tm


def brier_score(pmf: np.ndarray, realized_idx: int) -> float:
    """Multi-category Brier score: sum_k (p_k - 1{k=realized})^2."""
    indicator = np.zeros(len(pmf))
    indicator[realized_idx] = 1.0
    return float(np.sum((pmf - indicator) ** 2))


def cross_sectional_disp(pmf_matrix: np.ndarray) -> float:
    """Cross-sectional std of expected option index."""
    K = pmf_matrix.shape[1]
    means = pmf_matrix @ np.arange(K, dtype=float)
    return float(np.std(means, ddof=1))


# ── main backtest ─────────────────────────────────────────────────────────────

OPTION_LETTERS = list("abcdefghij")


def run_multicategory_backtest() -> pd.DataFrame:
    ifps = load_ifps()
    multi_ifps = ifps[ifps["n_opts"] >= 3].copy()
    outcome_map  = dict(zip(multi_ifps["ifp_id"], multi_ifps["outcome"]))
    K_map        = dict(zip(multi_ifps["ifp_id"], multi_ifps["n_opts"]))
    options_map  = dict(zip(multi_ifps["ifp_id"], multi_ifps["options"].fillna("")))
    multi_ids    = set(multi_ifps["ifp_id"])

    print("Loading GJP Years 1–4 forecasts...")
    all_fcasts = load_all_years()
    all_fcasts = all_fcasts[all_fcasts["ifp_id"].isin(multi_ids)].copy()
    all_fcasts["fcast_date"] = pd.to_datetime(all_fcasts["fcast_date"])

    # Last forecast per (user, question, option)
    last = (
        all_fcasts.sort_values("fcast_date")
        .groupby(["ifp_id", "user_id", "answer_option"])["value"]
        .last()
        .reset_index()
    )

    rows = []
    for qid, group in last.groupby("ifp_id"):
        K = K_map.get(qid)
        if K is None:
            continue
        outcome = outcome_map.get(qid)
        if pd.isna(outcome):
            continue

        expected_options = OPTION_LETTERS[:K]

        # Pivot to (user × option) matrix; keep only users with all K options
        pivot = group.pivot(index="user_id", columns="answer_option", values="value")
        pivot = pivot.reindex(columns=expected_options).dropna()
        if len(pivot) < MIN_FORECASTERS:
            continue

        pmf_matrix = pivot.values.astype(float)
        # Normalize rows to sum to 1
        row_sums = pmf_matrix.sum(axis=1, keepdims=True)
        pmf_matrix = pmf_matrix / np.where(row_sums > 0, row_sums, 1.0)

        # Realized bin index
        try:
            realized_idx = expected_options.index(str(outcome).strip().lower())
        except ValueError:
            continue

        am   = pmf_matrix.mean(axis=0)
        bc   = coord_median_pmf(pmf_matrix)
        tm   = trimmed_mean_pmf(pmf_matrix)
        disp = cross_sectional_disp(pmf_matrix)
        cat  = classify_question(options_map.get(qid, ""))

        rows.append({
            "qid":   qid,
            "K":     K,
            "cat":   cat,
            "bs_am": brier_score(am,  realized_idx),
            "bs_bc": brier_score(bc,  realized_idx),
            "bs_tm": brier_score(tm,  realized_idx),
            "disp":  disp,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "gjp_multicategory_backtest.csv", index=False)
    print(f"\nWrote {len(df)} questions → output/gjp/gjp_multicategory_backtest.csv")
    print(f"  K distribution: {dict(df.K.value_counts().sort_index())}")
    print()

    # ── summary by category ───────────────────────────────────────────────────
    for cat in ["unordered", "temporal", "ordinal"]:
        sub = df[df["cat"] == cat]
        if len(sub) == 0:
            continue
        wins_bc = (sub["bs_bc"] < sub["bs_am"]).mean() * 100
        wins_tm = (sub["bs_tm"] < sub["bs_am"]).mean() * 100
        red_bc  = (sub["bs_am"].mean() - sub["bs_bc"].mean()) / sub["bs_am"].mean() * 100
        red_tm  = (sub["bs_am"].mean() - sub["bs_tm"].mean()) / sub["bs_am"].mean() * 100
        print(f"  {cat:10s} n={len(sub):3d}  "
              f"BC wins {wins_bc:.0f}%  BS_BC {red_bc:+.1f}%  "
              f"TM wins {wins_tm:.0f}%  BS_TM {red_tm:+.1f}%  "
              f"AM={sub['bs_am'].mean():.4f}  BC={sub['bs_bc'].mean():.4f}  "
              f"TM={sub['bs_tm'].mean():.4f}")

    print()
    all_bc_wins = (df["bs_bc"] < df["bs_am"]).mean() * 100
    all_bc_red  = (df["bs_am"].mean() - df["bs_bc"].mean()) / df["bs_am"].mean() * 100
    print(f"  {'ALL':10s} n={len(df):3d}  "
          f"BC wins {all_bc_wins:.0f}%  BS_BC {all_bc_red:+.1f}%  "
          f"AM={df['bs_am'].mean():.4f}  BC={df['bs_bc'].mean():.4f}  "
          f"TM={df['bs_tm'].mean():.4f}")

    return df


if __name__ == "__main__":
    run_multicategory_backtest()
