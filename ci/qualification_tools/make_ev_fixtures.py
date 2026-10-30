#!/usr/bin/env python3
"""Create tiny DNACT inputs exposing EV/list/CSR consistency risks.

This generator runs no simulator and changes no project source. The fixtures
are valid according to the inspected parsers, even though some describe an
unusual backbone. Their purpose is either to establish numerical support or
to require an explicit fail-closed rejection before a release.
"""
from pathlib import Path
import json
import math
import os

ROOT = Path(os.environ.get('OXDNA_EV_FIXTURE_ROOT', str(Path(__file__).resolve().parent / 'fixtures')))

CASES = {
    'ordinary_chain': {
        'top': '2 1 0 2 0\n-1 A -1 1\n-1 A 0 -1\n',
        'par': '2\n0 1 0.34 s 0 1 -1 1 1\n',
        'positions': [(1, 1, 1), (1.34, 1, 1)],
        'contact': [0, 1], 'cpu_ev_count': 1, 'cuda_ev_count_predicted': 1,
    },
    'adjacent_csr_without_backbone': {
        'top': '2 1 0 2 0\n-1 A -1 -1\n-1 A -1 -1\n',
        'par': '2\n0 1 0.34 s 0 1 -1 1 1\n',
        'positions': [(1, 1, 1), (1.34, 1, 1)],
        'contact': [0, 1], 'cpu_ev_count': 1, 'cuda_ev_count_predicted': 2,
    },
    'nonadjacent_backbone': {
        'top': '3 1 0 3 0\n-1 A -1 2\n-1 A -1 -1\n-1 A 0 -1\n',
        # Two springs avoid the legacy one-spring/N>2 parser rejection.
        'par': '3\n0 1 2 s 0 1 -1 1 1\n0 2 0.34 s 0\n',
        'positions': [(1, 1, 1), (3, 1, 1), (1.34, 1, 1)],
        'contact': [0, 2], 'cpu_ev_count': 1, 'cuda_ev_count_predicted': 0,
    },
    'backbone_without_csr': {
        'top': '3 1 0 3 0\n-1 A -1 1\n-1 A 0 -1\n-1 A -1 -1\n',
        'par': '3\n0 2 2 s 0\n1 2 1.66 s 0 1 -1 1 1\n',
        'positions': [(1, 1, 1), (1.34, 1, 1), (3, 1, 1)],
        'contact': [0, 1], 'cpu_ev_count': 1, 'cuda_ev_count_predicted': 0,
    },
}

BASE = '''backend = {backend}
{cuda}interaction_type = DNACT
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
trajectory_file = trajectory.dat
energy_file = energy.dat
lastconf_file = last.dat
print_conf_interval = 1
print_energy_every = 1
time_scale = linear
restart_step_counter = 1
data_output_1 = {{
 name = pair_energy.dat
 print_every = 1
 col_1 = {{
  type = pair_energy
 }}
}}
'''

sigma, distance = 0.35, 0.34
lj6 = (sigma / distance)**6
one_ev = 4 * 2.0 * (lj6**2 - lj6)  # EXCL_EPS=2 in model.h:56
force_on_first_x = 24 * 2.0 * (lj6 - 2 * lj6**2) / distance

for name, case in CASES.items():
    folder = ROOT / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'topology.top').write_text(case['top'])
    (folder / 'network.par').write_text(case['par'])
    conf = 't = 0\nb = 20 20 20\nE = 0 0 0\n'
    for p in case['positions']:
        # x y z, a1, a3, translational velocity, body angular momentum.
        conf += ' '.join(map(str, (*p, 1, 0, 0, 0, 0, 1, 0, 0, 0, .1, .2, .3))) + '\n'
    (folder / 'initial.dat').write_text(conf)
    for backend in ('CPU', 'CUDA'):
        cuda = ('backend_precision = mixed\nCUDA_list = verlet\n'
                'CUDA_sort_every = 0\nuse_edge = 1\nedge_n_forces = 1\n'
                'CUDA_print_energy = 1\n') if backend == 'CUDA' else 'list_type = verlet\n'
        (folder / ('input_' + backend)).write_text(BASE.format(backend=backend, cuda=cuda))

report = {
    'status': 'STATIC_CODE_PATH_PREDICTION_NOT_GPU_MEASUREMENT',
    'distance': distance,
    'one_pair_ev_energy': one_ev,
    'one_pair_force_on_first_x': force_on_first_x,
    'cases': {name: {k: v for k, v in case.items()
                     if k not in ('top', 'par', 'positions')}
              for name, case in CASES.items()},
    'execution_note': 'Run native GPU diagnostics on the same snapshot, or reject unsupported topology before force evaluation. PairEnergy is host-side and alone cannot prove CUDA force/native energy parity.',
}
assert all(math.isfinite(x) for x in (one_ev, force_on_first_x))
(ROOT / 'expected.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
