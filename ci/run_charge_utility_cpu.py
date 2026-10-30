#!/usr/bin/env python3
"""Strict CPU-only charge utility gate; use the test's native CPU-only mode.

No modification of test AST, no CUDA execution, and explicit success/failure
markers are required for every CPU DNANM and DNACT fixture.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--script", type=Path, required=True)
parser.add_argument("--report", type=Path, required=True)
args = parser.parse_args()
script = args.script.resolve()
env = dict(os.environ, OXDNA_TEST_CPU_ONLY="1", PYTHONDONTWRITEBYTECODE="1")
run = subprocess.run(
    [sys.executable, "-B", str(script)],
    env=env, cwd=Path.cwd(), stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT, text=True, timeout=600, check=False,
)
print(run.stdout, end="", flush=True)
expected = {
    "CHARGE_PREPARATOR_VALIDATOR_GATE=PASS",
    "CUDA_RUNTIME_CASES=NOT_RUN_CPU_ONLY_MODE",
    "PREPARED_RUNTIME_CPU_DNANM=PASS",
    "PREPARED_RUNTIME_CPU_DNACT=PASS",
}
bad_labels = (
    "DUPLICATE", "MISSING", "WRONG_RESIDUE", "NAN",
    "OVERFLOW", "DUPLICATE_METADATA", "HEADER_EXTRA_TOKEN",
    "NPRO_TRAILING_TOKEN", "NGROUPS_TRAILING_TOKEN",
)
for mode in ("DNANM", "DNACT"):
    for label in bad_labels:
        expected.add(f"RUNTIME_REJECT_{label}_CPU_{mode}=PASS")
lines = set(run.stdout.splitlines())
missing = sorted(expected - lines)
cuda_executed = any("PREPARED_RUNTIME_CUDA_" in line or
                    ("RUNTIME_REJECT_" in line and "_CUDA_" in line)
                    for line in lines)
passed = run.returncode == 0 and not missing and not cuda_executed
report = {
    "passed": passed, "tests": 20 if passed else 0, "skipped": 0,
    "failures": 0 if passed else 1, "errors": 0,
    "exit_code": run.returncode, "backend": "CPU",
    "cuda_qualified": False, "missing_markers": missing,
    "unexpected_cuda_execution": cuda_executed,
    "case_markers_verified": len(expected) - len(missing),
}
args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
if not passed:
    print("CPU_CHARGE_UTILITY_GATE=FAIL", flush=True)
    raise SystemExit(1)
print("CPU_CHARGE_UTILITY_ADAPTED_GATE=PASS", flush=True)
