#!/usr/bin/env bash
# Run targeted PR #236 CPU regressions with strict fail-fast handling.
# All test commands must succeed; a single failure makes this script exit nonzero.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="${1:-$ROOT/build/bin/oxDNA}"
RESULTS_DIR="${2:-$ROOT/pr236_focused_results}"
if [[ "${BIN}" != /* ]]; then BIN="$(cd "$(dirname "$BIN")" && pwd)/$(basename "$BIN")"; fi
[[ -x "$BIN" ]] || { echo "ERROR: missing executable $BIN" >&2; exit 2; }
mkdir -p "$RESULTS_DIR"
echo "OXDNA_CPU_BINARY=$BIN"
echo "TARGETED_RESULTS=$RESULTS_DIR"
cd "$ROOT"

echo "=== 1/4 ANM energy and interchain springs ==="
python3 test/DNANM/audit_pr236_angular_energy.py --binary "$BIN" 2>&1 | tee "$RESULTS_DIR/01_angular_interchain.log"

echo "=== 2/4 PairForce and PairEnergy ==="
python3 test/DNANM/audit_pr236_pair_observables.py --binary "$BIN" --output "$RESULTS_DIR/pair_results.json" --pairenergy 2>&1 | tee "$RESULTS_DIR/02_pair_observables.log"

echo "=== 3/4 ANM numeric-base topology ==="
python3 test/DNANM/audit_pr236_topology_semantics.py --binary "$BIN" 2>&1 | tee "$RESULTS_DIR/03_numeric_topology.log"

echo "=== 4/4 ANMT finite decimal validation ==="
python3 test/DNANM/audit_pr236_parameter_validation.py --binary "$BIN" 2>&1 | tee "$RESULTS_DIR/04_parameter_validation.log"

echo "PR236_FOCUSED_CPU_ALL_FOUR_PASS"
