#!/usr/bin/env bash
# =============================================================================
# replicate.sh — full replication script for
#   "Robust Aggregation of Expert Probability Forecasts via Wasserstein Barycenters"
#
# Run from the repository root:
#   chmod +x replicate.sh
#   ./replicate.sh
#
# Requirements: Python 3.9+ with numpy, pandas, matplotlib, scipy, cvxpy, requests
#   pip install numpy pandas matplotlib scipy cvxpy requests
#
# Output: all figures written to figures/ and to the relevant output/ subdirectory.
# =============================================================================

set -e
PYTHON=${PYTHON:-python3}
EXAMPLES=python/examples
export PYTHONPATH=python

echo "============================================================"
echo " Wasserstein Barycenter — full replication"
echo "============================================================"

# ── 0. Sanity-check dependencies ─────────────────────────────────────────────
$PYTHON - <<'EOF'
import importlib, sys
missing = [m for m in ["numpy","pandas","matplotlib","scipy","cvxpy","requests"]
           if importlib.util.find_spec(m) is None]
if missing:
    print("ERROR: missing packages:", " ".join(missing))
    sys.exit(1)
print("Dependencies OK")
EOF

mkdir -p figures output/simulations output/gjp output/spf_application \
         output/spf_backtest output/ecb_spf_backtest output/application

# ── 1. Simulation figures (Section 3 & 4) ────────────────────────────────────
echo ""
echo "[1/9] Simulation robustness figures (fig2–fig5)..."
$PYTHON $EXAMPLES/robustness.py

echo "[2/9] Efficiency vs alpha figure (fig_efficiency_alpha)..."
$PYTHON $EXAMPLES/simulations_study.py

# ── 2. GJP binary evaluation (Section 5) ─────────────────────────────────────
echo ""
echo "[3/9] GJP Brier score figure (fig_gjp1_brier)..."
$PYTHON $EXAMPLES/gjp_figures.py

# ── 3. US SPF application — single-quarter illustration (Section 6) ──────────
echo ""
echo "[4/9] SPF single-quarter illustration (fig_spf1–3)..."
$PYTHON $EXAMPLES/spf_application.py

# ── 4. US SPF backtest — main backtest + reliability diagram (Section 6) ─────
echo ""
echo "[5/9] SPF full backtest + reliability diagram (fig_bt1–3, fig_bt8)..."
$PYTHON $EXAMPLES/spf_backtest.py

# ── 5. US SPF blend (Section 6) ──────────────────────────────────────────────
echo ""
echo "[6/9] SPF adaptive blend OOS evaluation..."
$PYTHON $EXAMPLES/spf_blend.py

# ── 6. US SPF blend figure + mass-correct-bin figure (fig_bt6, fig_bt7) ──────
echo ""
echo "[7/9] SPF blend figures (fig_bt6, fig_bt7)..."
$PYTHON $EXAMPLES/spf_backtest_blend_figure.py

# ── 7. ECB SPF backtest — computes ecb_backtest.csv (Section 6) ──────────────
echo ""
echo "[8/9] ECB SPF backtest (fetches live data from ECB SDW)..."
$PYTHON $EXAMPLES/ecb_spf_backtest.py

# ── 8. ECB figure (fig_ecb1) ─────────────────────────────────────────────────
echo ""
echo "[8b/9] ECB mass-correct-bin figure (fig_ecb1)..."
$PYTHON $EXAMPLES/ecb_spf_figures.py

# ── 9. Opioid application — Appendix A ───────────────────────────────────────
echo ""
echo "[9/9] Opioid application figures (fig_app1–4)..."
$PYTHON $EXAMPLES/application.py

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Replication complete."
echo " All figures written to figures/ and output/ subdirectories."
echo "============================================================"
$PYTHON - <<'EOF'
from pathlib import Path
figs = sorted(Path("figures").glob("*.pdf"))
print(f" {len(figs)} PDF figures in figures/:")
for f in figs:
    print(f"   {f.name}")
EOF
