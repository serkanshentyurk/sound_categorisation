#!/bin/bash
#
# Smoke-test runner for all scripts.
#
# Runs every entry point with --smoke-test to verify the pipeline
# works end-to-end before submitting to SLURM.
#
# Usage:
#   bash scripts/test_all.sh
#
# Exit codes:
#   0: all tests passed
#   1: one or more tests failed

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

PASS=0
FAIL=0
SKIP=0
RESULTS=()

run_test() {
    local name="$1"
    shift
    echo ""
    echo "=================================================="
    echo "  TEST: $name"
    echo "=================================================="
    echo "  CMD: $@"
    echo ""

    if "$@" 2>&1; then
        echo "  RESULT: PASS"
        RESULTS+=("PASS  $name")
        ((PASS++))
    else
        echo "  RESULT: FAIL"
        RESULTS+=("FAIL  $name")
        ((FAIL++))
    fi
}

skip_test() {
    local name="$1"
    local reason="$2"
    echo ""
    echo "  SKIP: $name ($reason)"
    RESULTS+=("SKIP  $name ($reason)")
    ((SKIP++))
}

# ─── Syntax checks ──────────────────────────────────────────────────────────
run_test "syntax: analysis_grid_search" \
    python3 -c "import ast; ast.parse(open('analysis/grid_search.py').read())"

run_test "syntax: inference_comparison" \
    python3 -c "import ast; ast.parse(open('inference/comparison.py').read())"

run_test "syntax: analysis_validation" \
    python3 -c "import ast; ast.parse(open('analysis/validation.py').read())"

run_test "syntax: scripts_config" \
    python3 -c "import ast; ast.parse(open('scripts/config.py').read())"

# ─── Imports ──────────────────────────────────────────────────────────────────
run_test "import: scripts.config" \
    python3 -c "from scripts.config import build_metadata; print(build_metadata('test', {})['hostname'])"

# ─── Generate synthetic cohort ────────────────────────────────────────────────
run_test "generate_synthetic_cohort (smoke)" \
    python3 scripts/generate_synthetic_cohort.py --smoke-test

# ─── Synthetic GS ────────────────────────────────────────────────────────────
run_test "synth_gs (smoke, animal 0, BE, UM)" \
    python3 scripts/validation/run_synth_gs.py \
        --cohort static_uniform --animal-index 0 --model BE \
        --fit-target update_matrix --smoke-test

run_test "synth_gs (smoke, animal 0, SC, UM)" \
    python3 scripts/validation/run_synth_gs.py \
        --cohort static_uniform --animal-index 0 --model SC \
        --fit-target update_matrix --smoke-test

run_test "synth_gs (smoke, animal 0, BE, CP)" \
    python3 scripts/validation/run_synth_gs.py \
        --cohort static_uniform --animal-index 0 --model BE \
        --fit-target conditional_psych --smoke-test

# ─── SNPE training ───────────────────────────────────────────────────────────
# Requires torch/sbi
if python3 -c "import torch; import sbi" 2>/dev/null; then
    run_test "train_snpe (smoke, BE uniform)" \
        python3 scripts/train_snpe.py --model be --distribution uniform --smoke-test

    run_test "train_snpe (smoke, SC uniform)" \
        python3 scripts/train_snpe.py --model sc --distribution uniform --smoke-test

    # ─── Synthetic SBI (after SNPE trained) ────────────────────────────────
    run_test "synth_sbi (smoke, animal 0, UM)" \
        python3 scripts/validation/run_synth_sbi.py \
            --cohort static_uniform --animal-index 0 \
            --fit-target update_matrix --smoke-test
else
    skip_test "train_snpe" "torch/sbi not installed"
    skip_test "synth_sbi" "torch/sbi not installed"
fi

# ─── Gather results ──────────────────────────────────────────────────────────
run_test "gather_cv_results (validation)" \
    python3 scripts/gather_cv_results.py --all --include-validation

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "=================================================="
echo "  SUMMARY"
echo "=================================================="
for r in "${RESULTS[@]}"; do
    echo "  $r"
done
echo ""
echo "  PASS: $PASS  FAIL: $FAIL  SKIP: $SKIP"
echo "=================================================="

if [ $FAIL -gt 0 ]; then
    exit 1
fi
exit 0
