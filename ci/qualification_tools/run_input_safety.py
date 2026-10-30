#!/usr/bin/env python3
"""Tiny CPU negative probes for global DNACT stiffness and MC2 box moves."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--binary', required=True)
p.add_argument('--work', required=True)
args = p.parse_args()
root = Path(args.work).resolve()
root.mkdir(parents=True, exist_ok=True)
os.environ['OXDNA_EV_FIXTURE_ROOT'] = str(root / 'fixtures')
spec = importlib.util.spec_from_file_location('ev_fixtures', Path(__file__).with_name('make_ev_fixtures.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
base = root / 'fixtures' / 'ordinary_chain'
original = (base / 'input_CPU').read_text()
probes = {
    'bending_nan': original.replace('bending_k = 45', 'bending_k = nan'),
    'torsion_inf': original.replace('torsion_k = 32', 'torsion_k = inf'),
    'both_nan': original.replace('bending_k = 45', 'bending_k = nan').replace('torsion_k = 32', 'torsion_k = nan'),
    'bending_malformed': original.replace('bending_k = 45', 'bending_k = garbage'),
    'native_dh_strength_inf': original + '\ndh_strength = inf\n',
    'MC2_volume': original.replace('sim_type = MD', 'sim_type = MC2') + '''
ensemble = NPT
P = 0
move_1 = {
 type = volume
 delta = 0.001
 prob = 1
 compute_energy_before = 1
}
''',
}
results = []
for name, inp in probes.items():
    folder = root / name
    folder.mkdir(exist_ok=True)
    for fn in ('topology.top', 'network.par', 'initial.dat'):
        (folder / fn).write_bytes((base / fn).read_bytes())
    (folder / 'input').write_text(inp)
    run = subprocess.run([str(Path(args.binary).resolve()), 'input'], cwd=folder,
                         text=True, capture_output=True, timeout=15)
    (folder / 'stdout.txt').write_text(run.stdout)
    (folder / 'stderr.txt').write_text(run.stderr)
    row = {'probe': name, 'exit_code': run.returncode,
           'status': 'REJECTED' if run.returncode else 'ACCEPTED',
           'error_tail': run.stderr[-1000:]}
    if (folder / 'last.dat').exists():
        lines = (folder / 'last.dat').read_text().splitlines()
        row['final_box'] = lines[1] if len(lines) > 1 else None
        row['box_changed'] = row['final_box'] != 'b = 20 20 20'
    if (folder / 'energy.dat').exists():
        energy = (folder / 'energy.dat').read_text()
        row['energy'] = energy
        numeric = []
        for token in energy.split():
            try:
                numeric.append(float(token))
            except ValueError:
                pass
        row['all_energy_numbers_finite'] = all(math.isfinite(v) for v in numeric)
    results.append(row)
report = {'test': 'DNACT_INPUT_SAFETY', 'results': results,
          'all_rejected': all(r['exit_code'] != 0 for r in results)}
(root / 'input_safety_report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
print(json.dumps(report, indent=2, allow_nan=False))
