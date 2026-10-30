#!/usr/bin/env python3
"""Bounded CPU/CUDA one-step impulse test for DNACT PP excluded volume.

The force reference is the scalar derivative of the specified LJ potential,
independent of both CPU and CUDA implementations. A CUDA fail-closed rejection
is recorded separately from a physics pass. No production source is changed.
"""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--binary', required=True)
p.add_argument('--work', required=True)
p.add_argument('--backends', default='CPU,CUDA')
args = p.parse_args()
binary = Path(args.binary).resolve()
root = Path(args.work).resolve()
root.mkdir(parents=True, exist_ok=True)
os.environ['OXDNA_EV_FIXTURE_ROOT'] = str(root / 'fixtures')
spec = importlib.util.spec_from_file_location('ev_fixtures', Path(__file__).with_name('make_ev_fixtures.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

env = os.environ.copy()
env['OXDNA_EXPERIMENTAL_DNACT_CUDA'] = 'RC2_GLOBAL_ANGULAR_RESEARCH_ONLY'
env.pop('OXDNA_EXPERIMENTAL_SORT_CUDA', None)
dt = 1e-6
expected_force = m.force_on_first_x
results = []
for name, case in m.CASES.items():
    for backend in args.backends.split(','):
        if backend not in ('CPU', 'CUDA'):
            raise ValueError(backend)
        folder = root / 'fixtures' / name / backend
        folder.mkdir(exist_ok=True)
        source = folder.parent
        for fn in ('topology.top', 'network.par', 'initial.dat'):
            (folder / fn).write_bytes((source / fn).read_bytes())
        (folder / 'input').write_bytes((source / ('input_' + backend)).read_bytes())
        run = subprocess.run([str(binary), 'input'], cwd=folder, env=env,
                             text=True, capture_output=True, timeout=30)
        (folder / 'stdout.txt').write_text(run.stdout)
        (folder / 'stderr.txt').write_text(run.stderr)
        row = {'case': name, 'backend': backend, 'exit_code': run.returncode,
               'expected_force_on_first_x': expected_force,
               'static_expected_cuda_ev_count': case['cuda_ev_count_predicted']}
        if run.returncode != 0:
            row['status'] = 'REJECTED' if 'ERROR:' in run.stderr else 'EXECUTION_FAILED'
            row['error_tail'] = run.stderr[-1500:]
        else:
            values = (folder / 'last.dat').read_text().splitlines()
            state = [list(map(float, line.split())) for line in values[3:] if line.strip()]
            assert len(state) == len(case['positions']) and all(len(v) == 15 for v in state)
            assert all(math.isfinite(v) for particle in state for v in particle)
            first = case['contact'][0]
            actual = state[first][9] / dt
            row['impulse_force_on_first_x'] = actual
            row['ev_force_multiple'] = actual / expected_force
            row['absolute_error'] = abs(actual - expected_force)
            row['force_tolerance'] = 5e-4 * abs(expected_force) + 1e-3
            row['status'] = 'PHYSICS_PASS' if row['absolute_error'] <= row['force_tolerance'] else 'PHYSICS_FAIL'
            native = re.findall(r'CUDA_energy:\s*([-+0-9.eE]+)',
                                run.stdout + '\n' + run.stderr + '\n' +
                                ((folder / 'energy.dat').read_text() if (folder / 'energy.dat').exists() else ''))
            row['native_cuda_energy_samples'] = [float(v) for v in native]
            row['energy_file'] = (folder / 'energy.dat').read_text() if (folder / 'energy.dat').exists() else ''
        results.append(row)
report = {'test': 'DNACT_EV_ONE_STEP_IMPULSE', 'dt': dt,
          'force_reference': '24*EXCL_EPS*((sigma/r)^6-2*(sigma/r)^12)/r, sigma=.35,r=.34,EXCL_EPS=2',
          'binary': str(binary), 'results': results,
          'all_executions_completed': all(r['status'] != 'EXECUTION_FAILED' for r in results),
          'scientific_all_pass': all(r['status'] == 'PHYSICS_PASS' for r in results),
          'note': 'One step verifies the initial force to O(dt), not long-time dynamics. A rejected topology is not a numerical pass.'}
(root / 'ev_impulse_report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
print(json.dumps(report, indent=2, allow_nan=False))
