"""
Dispersion-adaptive convex blend of arithmetic mean and W1 barycenter.

Model: Psi_t(lambda_t) = lambda_t * BC_t + (1 - lambda_t) * AM_t
       lambda_t = sigma(a + b * D_t)   [logistic in dispersion]

Evaluation: expanding-window out-of-sample (first 20 quarters as seed).
"""
import sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize_scalar, minimize
from scipy.special import expit   # logistic sigma


from spf_backtest import (
    load_all_quarters, cdf_median, rps, realized_bin, fetch_q4q4_cpi, BIN_UPPERS,
    trimmed_mean,
)

OUT_DIR = Path("output/spf_backtest")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── data ─────────────────────────────────────────────────────────────────────
panel = load_all_quarters("current")
try:
    cpi = fetch_q4q4_cpi("CPILFESL")   # SA
except RuntimeError as e:
    print(f"FRED download failed: {e}")
    print("Using hardcoded Q4/Q4 core CPI values (2007–2025).")
    hardcoded = {
        2007: 2.4, 2008: 1.8, 2009: 1.7, 2010: 0.8,
        2011: 2.2, 2012: 1.9, 2013: 1.7, 2014: 1.6,
        2015: 2.1, 2016: 2.2, 2017: 1.8, 2018: 2.2,
        2019: 2.3, 2020: 1.6, 2021: 5.0, 2022: 5.7,
        2023: 3.9, 2024: 3.2, 2025: 2.6,
    }
    cpi = pd.Series(hardcoded, name="CPILFESL_hardcoded")

records = []
for (yr, qt), pmfs in sorted(panel.items()):
    if yr not in cpi.index:
        continue
    val = cpi.loc[yr]
    if np.isnan(val):
        continue
    bin_idx = realized_bin(val)
    am = pmfs.mean(axis=0)
    bc = cdf_median(pmfs)
    tm = trimmed_mean(pmfs, trim=0.10)
    disp = float(np.var(pmfs @ np.arange(len(BIN_UPPERS))))   # variance of forecast means
    t = yr + (qt - 1) / 4
    records.append(dict(t=t, yr=yr, qt=qt, bin_idx=bin_idx,
                        am=am, bc=bc, tm=tm, disp=disp))

df = pd.DataFrame(records).sort_values("t").reset_index(drop=True)
N  = len(df)
print(f"Quarters: {N}")

# ── convenience ──────────────────────────────────────────────────────────────
def blend_rps(lam, row):
    pmf = lam * row["bc"] + (1 - lam) * row["am"]
    return rps(pmf, row["bin_idx"])

def mean_rps_fixed(lam, rows):
    return np.mean([blend_rps(lam, r) for r in rows])

# ── 1. FIXED LAMBDA: oracle (in-sample) ──────────────────────────────────────
lambdas = np.linspace(0, 1, 201)
rps_grid = [mean_rps_fixed(lam, df.to_dict("records")) for lam in lambdas]
lam_star  = lambdas[np.argmin(rps_grid)]
rps_star  = min(rps_grid)
rps_am    = mean_rps_fixed(0.0, df.to_dict("records"))
rps_bc    = mean_rps_fixed(1.0, df.to_dict("records"))
print(f"\nOracle fixed-lambda sweep:")
print(f"  AM  RPS = {rps_am:.4f}")
print(f"  BC  RPS = {rps_bc:.4f}")
print(f"  Best lambda* = {lam_star:.2f}  =>  RPS = {rps_star:.4f}")

# ── 2. DISPERSION-ADAPTIVE: expanding window OOS ─────────────────────────────
SEED = 20   # quarters in initial training window
rows_all = df.to_dict("records")

oos_results = []
for t_idx in range(SEED, N):
    train = rows_all[:t_idx]
    test  = rows_all[t_idx]
    D_test = test["disp"]

    # ── 2a. fixed lambda (expanding) ────────────────────────────────────────
    rps_grid_tr = [mean_rps_fixed(lam, train) for lam in lambdas]
    lam_fixed   = lambdas[np.argmin(rps_grid_tr)]
    rps_fixed   = blend_rps(lam_fixed, test)

    # ── 2b. dispersion-adaptive (logistic): tune a, b on training set ───────
    disps_tr = np.array([r["disp"] for r in train])
    D_std    = disps_tr.std() if disps_tr.std() > 0 else 1.0
    D_mean   = disps_tr.mean()

    def neg_rps_logistic(params):
        a, b = params
        total = 0.0
        for r in train:
            lam_t = expit(a + b * (r["disp"] - D_mean) / D_std)
            total += blend_rps(lam_t, r)
        return total / len(train)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimize(neg_rps_logistic, x0=[0.0, 0.0],
                       method="Nelder-Mead",
                       options={"xatol": 1e-4, "fatol": 1e-5, "maxiter": 2000})
    a_hat, b_hat = res.x
    lam_adaptive = expit(a_hat + b_hat * (D_test - D_mean) / D_std)
    rps_adaptive  = blend_rps(lam_adaptive, test)

    oos_results.append({
        "t":            test["t"],
        "bin_idx":      test["bin_idx"],
        "disp":         D_test,
        "rps_am":       blend_rps(0.0, test),
        "rps_bc":       blend_rps(1.0, test),
        "rps_tm":       rps(test["tm"], test["bin_idx"]),
        "lam_fixed":    lam_fixed,
        "rps_fixed":    rps_fixed,
        "lam_adaptive": lam_adaptive,
        "rps_adaptive": rps_adaptive,
    })

oos = pd.DataFrame(oos_results)
print(f"\nOOS evaluation (quarters {SEED}–{N-1}, n={len(oos)}):")
print(f"  AM         mean RPS = {oos['rps_am'].mean():.4f}")
print(f"  BC         mean RPS = {oos['rps_bc'].mean():.4f}")
print(f"  TM         mean RPS = {oos['rps_tm'].mean():.4f}")
print(f"  Fixed λ    mean RPS = {oos['rps_fixed'].mean():.4f}  (avg λ = {oos['lam_fixed'].mean():.2f})")
print(f"  Adaptive λ mean RPS = {oos['rps_adaptive'].mean():.4f}  (avg λ = {oos['lam_adaptive'].mean():.2f})")
rps_adapt = oos['rps_adaptive'].mean()
rps_am_oos = oos['rps_am'].mean()
rps_tm_oos = oos['rps_tm'].mean()
print(f"\nBlend vs AM  (OOS): {(rps_am_oos - rps_adapt)/rps_am_oos*100:.1f}%")
print(f"Blend vs TM  (OOS): {(rps_tm_oos - rps_adapt)/rps_tm_oos*100:.1f}%")

# ── Wilcoxon tests ────────────────────────────────────────────────────────────
from scipy.stats import wilcoxon
for col, label in [("rps_tm","TM"), ("rps_fixed","Fixed blend"), ("rps_adaptive","Adaptive blend")]:
    diff = oos["rps_am"] - oos[col]   # positive = col wins vs AM
    stat, p = wilcoxon(diff)
    wins = (diff > 0).mean()
    print(f"  {label}: wins {wins*100:.0f}% vs AM, Wilcoxon p={p:.4f}")

# ── blend vs TM ───────────────────────────────────────────────────────────────
diff_blend_tm = oos["rps_tm"] - oos["rps_adaptive"]   # positive = blend wins
stat, p_blend_tm = wilcoxon(diff_blend_tm)
wins_blend_tm = (diff_blend_tm > 0).mean()
print(f"\n  Adaptive blend vs TM: wins {wins_blend_tm*100:.0f}%, Wilcoxon p={p_blend_tm:.4f}")

# ── adaptive vs fixed blend ───────────────────────────────────────────────────
diff_adapt_fixed = oos["rps_fixed"] - oos["rps_adaptive"]   # positive = adaptive wins
stat, p_adapt_fixed = wilcoxon(diff_adapt_fixed)
wins_adapt_fixed = (diff_adapt_fixed > 0).mean()
rps_fixed_oos = oos["rps_fixed"].mean()
print(f"\n  Adaptive blend vs Fixed blend:")
print(f"    Adaptive RPS = {rps_adapt:.4f}, Fixed RPS = {rps_fixed_oos:.4f}")
print(f"    Adaptive wins {wins_adapt_fixed*100:.0f}% of quarters, Wilcoxon p={p_adapt_fixed:.4f}")

oos.to_csv(OUT_DIR / "blend_oos.csv", index=False)
print(f"\nSaved blend_oos.csv")
