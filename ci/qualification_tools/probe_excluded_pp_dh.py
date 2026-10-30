#!/usr/bin/env python3
"""Proposed one-step PP DH test: reciprocal topology link without CSR spring.

Only writes fixtures with --prepare-only. Execute explicitly with --binary;
the generator itself does not submit jobs or build/apply any project source.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--work', required=True)
p.add_argument('--binary')
p.add_argument('--backends', default='CPU,CUDA')
p.add_argument('--prepare-only', action='store_true')
args = p.parse_args()
root = Path(args.work).resolve()
root.mkdir(parents=True, exist_ok=True)
r, lam, strength, dt = .8, .3616455, .0543, 1e-6
reference_U = strength * math.exp(-r / lam) / r
reference_F0x = -strength * math.exp(-r / lam) * (1 / r**2 + 1 / (lam * r))
common = '''interaction_type = DNACT
topology = topology.top
parfile = network.par
conf_file = initial.dat
salt_concentration = 1
T = 300K
sim_type = MD
steps = 1
dt = 0.000001
thermostat = no
refresh_vel = 0
seed = 87611
bending_k = 45
torsion_k = 32
verlet_skin = 0.2
max_density_multiplier = 100
protein_charge_file = charges.dat
exclude_intragroup_electrostatics = 1
trajectory_file = trajectory.dat
energy_file = energy.dat
lastconf_file = last.dat
print_conf_interval = 1
print_energy_every = 1
time_scale = linear
restart_step_counter = 1
'''
top = '3 1 0 3 0\n-1 A -1 1\n-1 A 0 -1\n-1 A -1 -1\n'
par = '3\n0 2 2 s 0\n1 2 1.2 s 0 1 -1 1 1\n'
charges = '''# DNANM_ES_CHARGES 1
# index_space = protein_local
# index_base = 0
# charge_model = synthetic-unit-reduced-charge
# charge_source = independent-DH-regression
# n_protein_beads = 3
# n_groups = 3
# columns = protein_index topology_strand residue charge_e group_id
0 -1 A 0.815 0
1 -1 A 0.815 1
2 -1 A 0.0 2
'''
conf = 't = 0\nb = 20 20 20\nE = 0 0 0\n'
for x in (1., 1.8, 3.):
    conf += f'{x} 1 1 1 0 0 0 0 1 0 0 0 0.1 0.2 0.3\n'

cases = []
for backend in args.backends.split(','):
    assert backend in ('CPU', 'CUDA')
    for enabled in (False, True):
        folder = root / (backend + ('_ON' if enabled else '_OFF'))
        folder.mkdir(exist_ok=True)
        for name, data in {'topology.top': top, 'network.par': par, 'charges.dat': charges, 'initial.dat': conf}.items():
            (folder / name).write_text(data)
        specific = f'backend = {backend}\n'
        if backend == 'CUDA':
            specific += 'backend_precision = mixed\nCUDA_list = verlet\nCUDA_sort_every = 0\nuse_edge = 1\nedge_n_forces = 1\nCUDA_print_energy = 1\n'
        else:
            specific += 'list_type = verlet\n'
        (folder / 'input').write_text(specific + common + f'protein_protein_electrostatics = {int(enabled)}\n')
        cases.append((backend, enabled, folder))
report = {'test': 'DNACT_PPDH_EXCLUDED_LINK_WITHOUT_CSR',
          'status': 'PROPOSED_TEST_NOT_RUN', 'r': r, 'dt': dt,
          'reference_reduced_charges': [1., 1., 0.],
          'reference_pair_energy': reference_U,
          'reference_energy_per_particle': reference_U / 3,
          'reference_force_on_first_x': reference_F0x,
          'meaning': 'n3/n5=0-1 reciprocal; springs=0-2 and1-2; U_EV=0 because r=.8>PPcutoff; angular/spring initial responses zero; groups0!=1 allowed',
          'native_energy_tolerance': 2e-6,
          'force_tolerance': abs(reference_F0x) * 5e-4 + 1e-7,
          'results': []}
if not args.prepare_only:
    if not args.binary:
        p.error('--binary required unless --prepare-only')
    env = os.environ.copy()
    env['OXDNA_EXPERIMENTAL_DNACT_CUDA'] = 'RC2_GLOBAL_ANGULAR_RESEARCH_ONLY'
    for backend, enabled, folder in cases:
        run = subprocess.run([str(Path(args.binary).resolve()), 'input'], cwd=folder,
                             env=env, text=True, capture_output=True, timeout=30)
        (folder / 'stdout.txt').write_text(run.stdout)
        (folder / 'stderr.txt').write_text(run.stderr)
        row = {'backend': backend, 'pp_dh_enabled': enabled, 'exit_code': run.returncode}
        if run.returncode:
            row.update(status='REJECTED', error_tail=run.stderr[-1200:])
        else:
            state = [list(map(float, line.split())) for line in (folder / 'last.dat').read_text().splitlines()[3:] if line.strip()]
            assert len(state) == 3 and all(len(v) == 15 for v in state)
            assert all(math.isfinite(v) for particle in state for v in particle)
            measured = state[0][9] / dt
            expected = reference_F0x if enabled else 0.
            row.update(impulse_force_on_first_x=measured, expected_force=expected,
                       status='PHYSICS_PASS' if abs(measured-expected) <= report['force_tolerance'] else 'PHYSICS_FAIL')
            energies = (folder / 'energy.dat').read_text()
            row['energy_file'] = energies
            native = [float(v) for v in re.findall(r'CUDA_energy:\s*([-+0-9.eE]+)', run.stdout+'\n'+run.stderr+'\n'+energies)]
            row['native_cuda_energy_samples'] = native
            if backend == 'CUDA':
                expected_E = reference_U / 3 if enabled else 0.
                row['native_energy_pass'] = bool(native) and all(math.isfinite(v) and abs(v-expected_E) <= report['native_energy_tolerance'] for v in native)
                if not row['native_energy_pass']:
                    row['status'] = 'PHYSICS_FAIL'
        report['results'].append(row)
    report['status'] = 'PASS' if all(v['status'] == 'PHYSICS_PASS' for v in report['results']) else 'FAIL'
(root / 'excluded_pp_dh_report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
print(json.dumps(report, indent=2, allow_nan=False))
