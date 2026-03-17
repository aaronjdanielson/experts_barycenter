"""
Simulation study for Section 5 of the paper.

Produces:
  - Table 1: efficiency under null (no outliers)
  - Table 2: robustness under contamination
  - Figure: MSE ratio vs Dirichlet concentration alpha_0
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from wbarycenter import dw_barycenter

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "simulations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

K = 6
R = 200
RNG = np.random.default_rng(42)


def run_one(dat):
    n = dat.shape[0]
    weights = np.ones(n) / n
    D = np.ones((K, K)) - np.eye(K)
    am = dat.mean(axis=0)
    bc, _ = dw_barycenter(dat, weights, D)
    return am, bc


# ---------------------------------------------------------------------------
# Table 1: Efficiency under null (no outliers)
# ---------------------------------------------------------------------------

def table_efficiency():
    mu = np.ones(K) / K
    print("\n=== Table 1: Efficiency under null ===")
    print(f"{'N':>4}  {'AM MSE':>8}  {'BC MSE':>8}  {'BC/AM ratio':>11}")
    for N in [5, 10, 20]:
        am_mse_all, bc_mse_all = [], []
        for _ in range(R):
            dat = RNG.dirichlet(np.ones(K), size=N)
            am, bc = run_one(dat)
            am_mse_all.append(np.sum((am - mu) ** 2))
            bc_mse_all.append(np.sum((bc - mu) ** 2))
        am_mse = np.mean(am_mse_all)
        bc_mse = np.mean(bc_mse_all)
        print(f"{N:>4}  {am_mse:>8.4f}  {bc_mse:>8.4f}  {bc_mse/am_mse:>11.2f}")


# ---------------------------------------------------------------------------
# Table 2: Robustness under contamination
# ---------------------------------------------------------------------------

def table_robustness():
    alpha_out = np.array([1, 1, 1, 1, 1.5, 10.0]) ** 4
    N_reg = 10
    outlier_state = 5
    print("\n=== Table 2: Robustness under contamination ===")
    print(f"{'N_out':>5}  {'AM mean':>8}  {'AM sd':>6}  {'BC mean':>8}  {'BC sd':>6}")
    for N_out in [0, 1, 2, 3]:
        N_tot = N_reg + N_out
        am_s6, bc_s6 = [], []
        for _ in range(R):
            dat = np.vstack([
                RNG.dirichlet(np.ones(K), size=N_reg),
                *(RNG.dirichlet(alpha_out, size=1) for _ in range(N_out)),
            ])
            am, bc = run_one(dat)
            am_s6.append(am[outlier_state])
            bc_s6.append(bc[outlier_state])
        am_s6 = np.array(am_s6)
        bc_s6 = np.array(bc_s6)
        print(f"{N_out:>5}  {am_s6.mean():>8.3f}  {am_s6.std():>6.3f}"
              f"  {bc_s6.mean():>8.3f}  {bc_s6.std():>6.3f}")


# ---------------------------------------------------------------------------
# Figure: MSE ratio vs alpha_0
# ---------------------------------------------------------------------------

def fig_efficiency_alpha():
    mu = np.ones(K) / K
    alpha0_vals = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    N = 10
    ratios = []
    for alpha0 in alpha0_vals:
        alpha = np.ones(K) * alpha0
        am_mse_all, bc_mse_all = [], []
        for _ in range(R):
            dat = RNG.dirichlet(alpha, size=N)
            am, bc = run_one(dat)
            am_mse_all.append(np.sum((am - mu) ** 2))
            bc_mse_all.append(np.sum((bc - mu) ** 2))
        ratios.append(np.mean(bc_mse_all) / np.mean(am_mse_all))
        print(f"  alpha0={alpha0}: BC/AM ratio = {ratios[-1]:.3f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(alpha0_vals, ratios, "o-", color="steelblue", lw=2, ms=7)
    ax.axhline(1.0, color="gray", linestyle="--", lw=1, label="AM = BC")
    ax.set_xlabel(r"Dirichlet concentration $\alpha_0$")
    ax.set_ylabel("MSE ratio (BC / AM)")
    ax.set_title("Efficiency cost of barycenter vs. arithmetic mean")
    ax.set_xscale("log")
    max_ratio = max(ratios)
    ax.set_ylim(0.9, max_ratio * 1.08)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_efficiency_alpha.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig_efficiency_alpha.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved fig_efficiency_alpha to {OUTPUT_DIR}")


if __name__ == "__main__":
    table_efficiency()
    table_robustness()
    print("\nGenerating efficiency-alpha figure...")
    fig_efficiency_alpha()
