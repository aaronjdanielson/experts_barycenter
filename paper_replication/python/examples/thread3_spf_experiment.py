"""
Thread 3: Learned-Geometry Wasserstein Barycenter — SPF Experiment
===================================================================

Trains the LearnedWassersteinBarycenter model on US SPF PRCCPI current-year
histograms (K=10 inflation bins, 2007:Q1–2025:Q4, T=76 full-backtest quarters) using
walk-forward cross-validation.

Evaluations
-----------
1. OOS RPS comparison: AM | BC | TM | Blend | LGB(d=1) | LGB(d=2) | LGB+weights
2. Geometry visualization: learned φ_θ vs. natural spacing
3. Dispersion-conditional analysis
4. Learned cost matrix heatmap

Interpreter: /Users/aarondanielson/miniforge3/envs/supercluster/bin/python3.11
"""

import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from copy import deepcopy
from scipy.stats import wilcoxon

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "python"))

from wbarycenter.learned_geometry import (
    LearnedWassersteinBarycenter,
    QuantileLevelBarycenter,
    rps_loss,
    geometric_regularization,
    predict_numpy,
    predict_qlevel_numpy,
    get_cost_matrix_numpy,
    get_embedding_numpy,
)

# Reuse existing data loading from spf_backtest
from examples.spf_backtest import load_all_quarters, BIN_MIDPOINTS, K

OUT_DIR = ROOT / "output" / "thread3"
FIG_DIR = ROOT.parent / "figures"  # repo root figures/
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Model configs to compare ─────────────────────────────────────────────────
CONFIGS = {
    "LGB_d1":      dict(d=1, monotone=True,  use_weights=False),
    "LGB_d2":      dict(d=2, monotone=False, use_weights=False),
    "LGB_d1_W":    dict(d=1, monotone=True,  use_weights=True),
}

TRAIN_KWARGS = dict(
    n_epochs=400,
    lr=2e-3,
    lambda_smooth=0.1,
    lambda_natural=1.0,   # anchor geometry to natural spacing
    patience=50,
)

# Walk-forward split: train on 2007–2016, test on 2017–2024
TRAIN_END_YEAR = 2016
OOS_START = (2017, 1)
OOS_END_YEAR  = 2024   # explicit cap; SPF Table 3 covers 2007–2025 (76Q)

# If True: retrain each OOS quarter on all data up to t-1 (expanding window)
# This matches the real-time protocol but is slower (32 retraining runs)
EXPANDING_WINDOW = False

# ── Baseline aggregators (from spf_backtest) ─────────────────────────────────
from examples.spf_backtest import cdf_median, trimmed_mean

def arithmetic_mean(pmfs: np.ndarray) -> np.ndarray:
    return pmfs.mean(axis=0)

def bc_cdf(pmfs: np.ndarray) -> np.ndarray:
    return cdf_median(pmfs)

def tm_baseline(pmfs: np.ndarray) -> np.ndarray:
    return trimmed_mean(pmfs, trim=0.10)

def rps_numpy(pmf: np.ndarray, bin_idx: int) -> float:
    cdf_f = np.cumsum(pmf)[:-1]
    cdf_r = np.zeros(K - 1)
    cdf_r[bin_idx:] = 1.0
    return float(np.sum((cdf_f - cdf_r) ** 2))

def cross_sectional_std(pmfs: np.ndarray) -> float:
    means = pmfs @ BIN_MIDPOINTS
    return float(np.std(means, ddof=1))


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(
    train_data: list,   # list of (panel_np, bin_idx)
    val_data: list,
    config: dict,
    n_epochs: int = 400,
    lr: float = 2e-3,
    lambda_smooth: float = 0.1,
    lambda_natural: float = 1.0,
    patience: int = 50,
    device: str = "cpu",
) -> LearnedWassersteinBarycenter:
    """
    Train LearnedWassersteinBarycenter on a list of (panel, realized_bin) pairs.
    Uses early stopping on validation RPS.

    Returns best model (lowest val RPS).
    """
    model = LearnedWassersteinBarycenter(K=K, **config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_val_rps = float("inf")
    best_state = deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        for panel_np, y_bin in train_data:
            panel_t = torch.from_numpy(panel_np.astype(np.float32)).to(device)
            optimizer.zero_grad()
            pmf = model(panel_t)
            loss = rps_loss(pmf, y_bin, K)
            reg = geometric_regularization(model, lambda_smooth, lambda_natural)
            (loss + reg).backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        # Validation
        val_rps = evaluate_model(model, val_data, device)
        if val_rps < best_val_rps - 1e-6:
            best_val_rps = val_rps
            best_state = deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1:3d} | train_rps {train_loss/len(train_data):.4f}"
                  f" | val_rps {val_rps:.4f} | best {best_val_rps:.4f}")

    model.load_state_dict(best_state)
    return model


def evaluate_model(
    model: LearnedWassersteinBarycenter,
    data: list,       # list of (panel_np, bin_idx)
    device: str = "cpu",
) -> float:
    """Mean RPS on a list of (panel, bin_idx) pairs."""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for panel_np, y_bin in data:
            panel_t = torch.from_numpy(panel_np.astype(np.float32)).to(device)
            pmf = model(panel_t)
            total += rps_loss(pmf, y_bin, K).item()
    return total / len(data)


# ── Q-Level model training ────────────────────────────────────────────────────

def train_qlevel_model(
    train_data_disp: list,   # list of (panel_np, bin_idx, dispersion)
    n_epochs: int = 600,
    lr: float = 0.05,
    device: str = "cpu",
) -> QuantileLevelBarycenter:
    """Train QuantileLevelBarycenter on (panel, bin_idx, dispersion) triples."""
    model = QuantileLevelBarycenter(use_dispersion=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    no_improve = 0

    for epoch in range(n_epochs):
        model.train()
        total = 0.0
        for panel_np, y_bin, disp in train_data_disp:
            panel_t = torch.from_numpy(panel_np.astype(np.float32)).to(device)
            optimizer.zero_grad()
            pmf = model(panel_t, dispersion=disp)
            loss = rps_loss(pmf, y_bin, K)
            loss.backward()
            optimizer.step()
            total += loss.item()

        avg = total / len(train_data_disp)
        if avg < best_loss - 1e-6:
            best_loss = avg
            best_state = deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= 80:
            break

    model.load_state_dict(best_state)
    return model


# ── Walk-forward training + OOS evaluation ───────────────────────────────────

def run_experiment():
    print("Loading SPF panel data...")
    panel_dict = load_all_quarters(horizon="current")

    # ── Build ordered list of (year, qtr) with realized bin ─────────────────
    # Realized bins come from backtest_current.csv (already computed)
    backtest_path = ROOT / "output" / "spf_backtest" / "backtest_current.csv"
    bt = pd.read_csv(backtest_path)
    # Use SA series, one row per (year, qtr)
    bt_sa = bt[bt["series"] == "SA"].set_index(["year", "qtr"])

    quarters = sorted(panel_dict.keys())
    records = []
    for (yr, qt) in quarters:
        key = (yr, qt)
        if key not in bt_sa.index:
            continue
        row = bt_sa.loc[key]
        records.append({
            "year": yr,
            "qtr": qt,
            "bin_idx": int(row["bin_idx"]),
            "panel": panel_dict[key],
            "disp": cross_sectional_std(panel_dict[key]),
            "am_rps": float(row["am_rps"]),
            "bc_rps": float(row["bc_rps"]),
            "tm_rps": float(row["tm_rps"]),
        })

    print(f"Total quarters with both panel and realized: {len(records)}")

    # Split
    train_records = [r for r in records if r["year"] <= TRAIN_END_YEAR]
    oos_records   = [r for r in records if (r["year"], r["qtr"]) >= OOS_START
                     and r["year"] <= OOS_END_YEAR]
    print(f"Train: {len(train_records)} quarters | OOS: {len(oos_records)} quarters")

    # Format for training
    train_data = [(r["panel"], r["bin_idx"]) for r in train_records]
    train_data_disp = [(r["panel"], r["bin_idx"], r["disp"]) for r in train_records]
    val_data   = train_data[-8:]

    # ── Train Q-Level model (simple, 2 params) ───────────────────────────────
    print("\nTraining Q-Level model (simple, 2 params)...")
    ql_model = train_qlevel_model(train_data_disp)
    print(f"  α={ql_model.alpha.item():.3f}, β={ql_model.beta.item():.3f}")
    print(f"  q(D=0)={ql_model.quantile_level(0.0).item():.4f}, "
          f"q(D=0.5)={ql_model.quantile_level(0.5).item():.4f}")

    # ── Train full LGB configs (Sinkhorn, heavier) ───────────────────────────
    trained_models = {}
    for name, config in CONFIGS.items():
        print(f"\nTraining {name}...")
        model = train_model(
            train_data=train_data,
            val_data=val_data,
            config=config,
            **TRAIN_KWARGS,
        )
        trained_models[name] = model

    # ── OOS evaluation ───────────────────────────────────────────────────────
    oos_results = []
    for r in oos_records:
        panel_np = r["panel"]
        y_bin = r["bin_idx"]
        disp = r["disp"]

        row_out = {
            "year": r["year"],
            "qtr": r["qtr"],
            "bin_idx": y_bin,
            "disp": disp,
            "AM":  rps_numpy(arithmetic_mean(panel_np), y_bin),
            "BC":  r["bc_rps"],
            "TM":  r["tm_rps"],
            "QLevel": rps_numpy(predict_qlevel_numpy(ql_model, panel_np, disp), y_bin),
        }

        for name, model in trained_models.items():
            pmf = predict_numpy(model, panel_np)
            row_out[name] = rps_numpy(pmf, y_bin)

        oos_results.append(row_out)

    df = pd.DataFrame(oos_results)
    df.to_csv(OUT_DIR / "oos_results.csv", index=False)
    print(f"\nOOS results saved: {OUT_DIR / 'oos_results.csv'}")

    # ── Summary table ─────────────────────────────────────────────────────────
    methods = ["AM", "BC", "TM", "QLevel"] + list(CONFIGS.keys())
    print("\n" + "="*60)
    print(f"{'Method':<15} {'Mean RPS':>10} {'vs AM (%)':>10} {'vs BC (%)':>10} {'DM vs AM':>9}")
    print("-"*60)
    am_rps = df["AM"].values
    bc_rps = df["BC"].values
    for m in methods:
        if m not in df.columns:
            continue
        v = df[m].values
        pct_am = (v.mean() / am_rps.mean() - 1) * 100
        pct_bc = (v.mean() / bc_rps.mean() - 1) * 100
        if m == "AM":
            p_str = "—"
        else:
            try:
                _, p = wilcoxon(v, am_rps)
                p_str = f"{p:.3f}"
            except Exception:
                p_str = "—"
        print(f"{m:<15} {v.mean():>10.4f} {pct_am:>+9.1f}% {pct_bc:>+9.1f}% {p_str:>9}")
    print("="*60)

    # ── DM test: QLevel vs BC ────────────────────────────────────────────────
    print("\nDM test: QLevel vs BC:")
    try:
        _, p = wilcoxon(df["QLevel"].values, df["BC"].values)
        pct = (df["QLevel"].mean() / df["BC"].mean() - 1) * 100
        print(f"  QLevel={df['QLevel'].mean():.4f}, BC={df['BC'].mean():.4f}, "
              f"Δ={pct:+.1f}%, Wilcoxon p={p:.3f}")
    except Exception as e:
        print(f"  {e}")

    # ── Dispersion analysis ───────────────────────────────────────────────────
    median_disp = df["disp"].median()
    lo = df[df["disp"] <= median_disp]
    hi = df[df["disp"] > median_disp]
    print(f"\nDispersion-conditional (median split at D*={median_disp:.3f}):")
    print(f"  {'Method':<12} {'Lo-D':>8} {'Hi-D':>8} {'Lo-D vs BC':>12} {'Hi-D vs BC':>12}")
    for m in ["AM", "BC", "TM", "QLevel"]:
        if m in df.columns:
            lo_v = lo[m].mean()
            hi_v = hi[m].mean()
            lo_pct = (lo_v / lo["BC"].mean() - 1) * 100
            hi_pct = (hi_v / hi["BC"].mean() - 1) * 100
            print(f"  {m:<12} {lo_v:>8.4f} {hi_v:>8.4f} {lo_pct:>+11.1f}% {hi_pct:>+11.1f}%")

    # ── Make figures ─────────────────────────────────────────────────────────
    make_figures(df, trained_models, ql_model)

    return df, trained_models, ql_model


# ── Figures ───────────────────────────────────────────────────────────────────

BIN_LABELS = [
    "Decline", "0.0–0.4", "0.5–0.9", "1.0–1.4", "1.5–1.9",
    "2.0–2.4", "2.5–2.9", "3.0–3.4", "3.5–3.9", "≥4.0",
]

def make_figures(df: pd.DataFrame, trained_models: dict,
                 ql_model: "QuantileLevelBarycenter"):
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    colors = {"AM": "#888888", "BC": "#2196F3", "TM": "#FF9800",
              "QLevel": "#E91E63", "LGB_d1": "#4CAF50", "LGB_d2": "#9C27B0"}

    # ── Plot 1: Learned quantile level q(D) ─────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    d_range = np.linspace(0.0, 1.0, 200)
    import torch as _torch
    with _torch.no_grad():
        q_vals = [ql_model.quantile_level(float(d)).item() for d in d_range]
    ax1.plot(d_range, q_vals, color=colors["QLevel"], linewidth=2)
    ax1.axhline(0.5, color="black", linewidth=0.8, linestyle="--", label="q=0.5 (BC)")
    # Mark training dispersion range
    d_min, d_max = df["disp"].min(), df["disp"].max()
    ax1.axvspan(d_min, d_max, alpha=0.1, color="gray", label="OOS range")
    ax1.set_xlabel("Cross-sectional dispersion D_t", fontsize=8)
    ax1.set_ylabel("Quantile level q_θ(D)", fontsize=8)
    ax1.set_title("Learned quantile level vs. dispersion", fontsize=9, fontweight="bold")
    ax1.set_ylim(0, 1)
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    # ── Plot 2: OOS RPS over time ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    df_sorted = df.sort_values(["year", "qtr"]).copy()
    df_sorted["t"] = range(len(df_sorted))
    time_labels = df_sorted.apply(lambda r: f"{int(r.year)}Q{int(r.qtr)}", axis=1)

    for m in ["AM", "BC", "QLevel"]:
        if m in df_sorted.columns:
            s = df_sorted[m].rolling(4, min_periods=1).mean()
            ax2.plot(df_sorted["t"], s, label=m, color=colors.get(m, "#333333"),
                     linewidth=1.5)

    tick_idx = list(range(0, len(df_sorted), 4))
    ax2.set_xticks(tick_idx)
    ax2.set_xticklabels([time_labels.iloc[i] for i in tick_idx],
                        rotation=45, ha="right", fontsize=6)
    ax2.set_ylabel("RPS (4-qtr rolling mean)", fontsize=8)
    ax2.set_title("OOS RPS over time", fontsize=9, fontweight="bold")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    # ── Plot 3: Mean RPS bar chart ─────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    methods_bar = ["AM", "BC", "TM", "QLevel"] + [k for k in CONFIGS.keys() if k in df.columns]
    valid_methods = [m for m in methods_bar if m in df.columns]
    means = [df[m].mean() for m in valid_methods]
    cols_bar = [colors.get(m, "#666666") for m in valid_methods]
    bars = ax3.bar(valid_methods, means, color=cols_bar, edgecolor="black", linewidth=0.5)
    ax3.set_ylabel("Mean OOS RPS", fontsize=8)
    ax3.set_title("Mean RPS by method", fontsize=9, fontweight="bold")
    ax3.set_xticklabels(valid_methods, rotation=30, ha="right", fontsize=7)
    for bar, val in zip(bars, means):
        ax3.text(bar.get_x() + bar.get_width()/2, val + 0.002,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=6)
    ax3.grid(True, alpha=0.3, axis="y")

    # ── Plot 4: Δ RPS (QLevel - BC) vs dispersion ─────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    delta_ql = df["QLevel"].values - df["BC"].values
    ax4.scatter(df["disp"].values, delta_ql, alpha=0.7, s=25,
                color=colors["QLevel"], edgecolors="none", label="QLevel – BC")
    ax4.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax4.axvline(df["disp"].median(), color="red", linewidth=0.8, linestyle="--",
                alpha=0.7, label="Median D")
    m_reg = np.polyfit(df["disp"].values, delta_ql, 1)
    x_line = np.linspace(df["disp"].min(), df["disp"].max(), 100)
    ax4.plot(x_line, np.polyval(m_reg, x_line), color="black",
             linewidth=1.5, label=f"slope={m_reg[0]:+.3f}")
    ax4.set_xlabel("Dispersion D_t", fontsize=8)
    ax4.set_ylabel("Δ RPS (QLevel – BC)", fontsize=8)
    ax4.set_title("QLevel advantage vs. dispersion", fontsize=9, fontweight="bold")
    ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.3)

    # ── Plot 5: Learned bin geometry for LGB models ───────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    natural = np.linspace(0, 1, K)
    natural = natural - natural.mean()
    natural = natural / np.sqrt(np.mean(natural**2))
    ax5.plot(range(K), natural, 's--', label="Natural (=BC)", color="black",
             linewidth=1.5, markersize=4, alpha=0.7)
    for name, model in trained_models.items():
        if hasattr(model, "embedding") and model.embedding.d == 1:
            phi = get_embedding_numpy(model).squeeze()
            ax5.plot(range(K), phi, 'o-', label=name, color=colors.get(name),
                     linewidth=1.5, markersize=4)
    ax5.set_xticks(range(K))
    ax5.set_xticklabels(BIN_LABELS, rotation=45, ha="right", fontsize=7)
    ax5.set_ylabel("φ_θ(b)", fontsize=9)
    ax5.set_title("Learned bin geometry (LGB)", fontsize=9, fontweight="bold")
    ax5.legend(fontsize=7)
    ax5.grid(True, alpha=0.3)

    # ── Plot 6: QLevel vs BC scatter ──────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    if "QLevel" in df.columns:
        ql_vals = df["QLevel"].values
        bc_vals = df["BC"].values
        ax6.scatter(bc_vals, ql_vals, alpha=0.6, s=20,
                    color=colors["QLevel"], edgecolors="none")
        lo, hi_ = min(bc_vals.min(), ql_vals.min()), max(bc_vals.max(), ql_vals.max())
        ax6.plot([lo, hi_], [lo, hi_], "k--", linewidth=0.8, label="y=x (tie)")
        ax6.set_xlabel("BC RPS", fontsize=8)
        ax6.set_ylabel("Q-Level RPS", fontsize=8)
        ax6.set_title("Quarter-by-quarter: Q-Level vs BC", fontsize=9, fontweight="bold")
        # Count wins
        n_win = (ql_vals < bc_vals).sum()
        ax6.text(0.05, 0.92, f"QLevel wins {n_win}/{len(ql_vals)} quarters",
                 transform=ax6.transAxes, fontsize=7)
        ax6.legend(fontsize=7)
        ax6.grid(True, alpha=0.3)

    fig.suptitle("Thread 3: Learned-Geometry Wasserstein Barycenter — SPF Results",
                 fontsize=11, fontweight="bold")

    out_path = FIG_DIR / "fig_thread3_spf.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {out_path}")


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Quick run: 50 epochs, d=1 only")
    args = parser.parse_args()

    if args.quick:
        TRAIN_KWARGS["n_epochs"] = 50
        TRAIN_KWARGS["patience"] = 20
        for k in list(CONFIGS.keys()):
            if k != "LGB_d1":
                del CONFIGS[k]
        print("Quick mode: 50 epochs, LGB_d1 only")

    df, models, ql = run_experiment()
    print("\nDone.")
