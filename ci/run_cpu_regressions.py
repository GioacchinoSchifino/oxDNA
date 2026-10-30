#!/usr/bin/env python3
"""Run seven CPU unittest suites plus an explicit CPU utility adaptation.

This proposed CI driver does not qualify CUDA or launch GPU jobs.
Empty/skipped suites fail. Each unittest file runs in a separate process.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest

PROGRAMS = (
    "test_source_contract.py",
    "test_analytical_physics_core.py",
    "test_pairenergy_term_registration.py",
    "test_stress_tensor.py",
    "test_dnact_cpu_nonzero_coexistence.py",
    "test_surface_contract.py",
    "test_fail_closed_matrix.py",
    "test_valid_charge_runtime.py",
    "test_charge_utility_runtime.py",
)


def one(path, report):
    spec = importlib.util.spec_from_file_location("cpu_ci_case", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    data = dict(tests=result.testsRun, failures=len(result.failures),
                errors=len(result.errors), skipped=len(result.skipped),
                unexpected_successes=len(result.unexpectedSuccesses))
    data["passed"] = (result.wasSuccessful() and result.testsRun > 0
                      and not result.skipped)
    report.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    return 0 if data["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--output", type=Path, default=Path("ci-results"))
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--one", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--report", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.one:
        return one(args.one, args.report)
    if not args.source or not args.binary:
        parser.error("--source and --binary are required")
    source, binary = args.source.resolve(), args.binary.resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        parser.error("binary is missing or not executable")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OXDNA_REPO=str(source), OXDNA_BIN=str(binary),
               PYTHONDONTWRITEBYTECODE="1")
    for key in ("OXDNA_EXPERIMENTAL_DNACT_CUDA", "OXDNA_EXPERIMENTAL_SORT_CUDA",
                "OXDNA_EXPECT_CUDA_FLOAT"):
        env.pop(key, None)
    results = []
    for name in PROGRAMS:
        path = source / "test/DNANM/protein_electrostatics" / name
        report, log = output / (name + ".json"), output / (name + ".log")
        report.unlink(missing_ok=True)
        started = time.monotonic()
        item = dict(program=name, passed=False)
        try:
            with log.open("w") as stream:
                if name == 'test_charge_utility_runtime.py':
                    command = [sys.executable, '-B',
                               str(Path(__file__).with_name('run_charge_utility_cpu.py')),
                               '--script', str(path), '--report', str(report)]
                else:
                    command = [sys.executable, "-B", str(Path(__file__).resolve()),
                               "--one", str(path), "--report", str(report)]
                run = subprocess.run(
                    command,
                    env=env, cwd=source, stdout=stream, stderr=subprocess.STDOUT,
                    timeout=args.timeout, check=False)
            if report.exists():
                item.update(json.loads(report.read_text()))
            item["exit_code"] = run.returncode
            item["passed"] = item["passed"] and run.returncode == 0
            if name == 'test_charge_utility_runtime.py':
                output_text = log.read_text()
                item['passed'] = (item['passed']
                    and 'CPU_CHARGE_UTILITY_ADAPTED_GATE=PASS' in output_text
                    and 'CHARGE_PREPARATOR_VALIDATOR_GATE=PASS' in output_text
                    and 'PREPARED_RUNTIME_CUDA' not in output_text)
        except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
            item["error"] = repr(exc)
        item["seconds"] = time.monotonic() - started
        results.append(item)
        print(name, "PASS" if item["passed"] else "FAIL", flush=True)
    data = dict(programs=results, passed=sum(x["passed"] for x in results),
                total=len(results), backend="CPU", cuda_qualified=False)
    (output / "summary.json").write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n")
    return 0 if all(x["passed"] for x in results) else 1


if __name__ == "__main__":
    sys.exit(main())
