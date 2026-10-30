#!/usr/bin/env python3

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


OXDNA = Path(os.environ["OXDNA_BIN"]).resolve()


TOP = """4 4 1 3 1
1 A -1 -1
-1 K -1 -1
-2 K -1 -1
-3 K -1 -1
"""

CONF = """t = 0
b = 20 20 20
E = 0 0 0
5 5 5 1 0 0 0 0 1 0 0 0 1e-12 -2e-12 3e-12
4.66 5.3408 6.02 1 0 0 0 0 1 0 0 0 1e-12 -2e-12 3e-12
12 12 12 1 0 0 0 0 1 0 0 0 1e-12 -2e-12 3e-12
5.46 5.3408 6.02 1 0 0 0 0 1 0 0 0 1e-12 -2e-12 3e-12
"""

PAR = """3
1 2 12.2065556157 s 0.0
2 3 2.0000000000 s 0.0
"""


def charges(q0, q1, q2):
    return f"""# DNANM_ES_CHARGES 1
# index_space = protein_local
# index_base = 0
# charge_model = pairenergy-registration-test
# charge_source = analytical-test
# n_protein_beads = 3
# n_groups = 3
# columns = protein_index topology_strand residue charge_e group_id
0 -1 K {q0:.8f} 0
1 -2 K {q1:.8f} 1
2 -3 K {q2:.8f} 2
"""


def input_text(pd, pp, pair, print_header=False):
    return f"""backend = CPU
backend_precision = double
interaction_type = DNANM

sim_type = MD
T = 300K
dt = 0.002

thermostat = no
refresh_vel = 1
seed = 19

max_io = 16
verlet_skin = 0.25

ensemble = NVT
external_forces = 0

salt_concentration = 1.0
dh_half_charged_ends = false

parfile = anm.par
topology = system.top
conf_file = init.dat

trajectory_file = trajectory.dat
lastconf_file = last_conf.dat
energy_file = energy.dat
log_file = log.dat

print_conf_interval = 1
print_energy_every = 1
steps = 1

restart_step_counter = 1
time_scale = linear

protein_dna_electrostatics = {"true" if pd else "false"}
protein_protein_electrostatics = {"true" if pp else "false"}

protein_charge_file = protein_charges.dat
protein_charge_scale = 1.0
protein_dna_dh_scale = 1.0
protein_protein_dh_scale = 1.0
dna_effective_charge_e = 0.815
exclude_intragroup_electrostatics = true

data_output_1 = {{
 name = pair_energy.dat
 print_every = 1
 col_1 = {{
  type = pair_energy
  particle1_id = {pair[0]}
  particle2_id = {pair[1]}
  print_header = {"true" if print_header else "false"}
 }}
}}
"""


def first_numeric(path):
    for line in path.read_text().splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            return [float(x) for x in s.split()]
    raise RuntimeError(f"No numeric row in {path}")


def pairenergy_header(path):
    for line in path.read_text().splitlines():
        s = line.strip()
        if s.startswith("#id1 id2"):
            return s
    raise RuntimeError(f"No PairEnergy header in {path}")


def run_case(root, name, pd, pp, q, pair, print_header=False):
    case = root / name
    case.mkdir()

    (case / "system.top").write_text(TOP)
    (case / "init.dat").write_text(CONF)
    (case / "anm.par").write_text(PAR)
    (case / "protein_charges.dat").write_text(charges(*q))
    (case / "input").write_text(input_text(pd, pp, pair, print_header))

    proc = subprocess.run(
        [str(OXDNA), "input"],
        cwd=case,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
    )

    log = ""
    if (case / "log.dat").exists():
        log = (case / "log.dat").read_text(errors="replace")

    if proc.returncode != 0:
        raise RuntimeError(
            f"{name} failed\n{proc.stdout}\n{log}"
        )

    # energy.dat:
    # time potential kinetic total
    system_potential = first_numeric(
        case / "energy.dat"
    )[1]

    # PairEnergy: last value is total pair energy.
    pair_energy = first_numeric(
        case / "pair_energy.dat"
    )[-1]

    return system_potential, pair_energy


class PairEnergyTermRegistration(unittest.TestCase):

    def delta_check(self, label, pd, pp, q, pair):
        with tempfile.TemporaryDirectory(
            prefix="oxdna_pairenergy_reg_"
        ) as td:
            root = Path(td)

            sys_off, pair_off = run_case(
                root,
                label + "_off",
                False,
                False,
                q,
                pair,
            )

            sys_on, pair_on = run_case(
                root,
                label + "_on",
                pd,
                pp,
                q,
                pair,
            )

            d_system = sys_on - sys_off
            d_pair = pair_on - pair_off

            self.assertGreater(
                abs(d_system),
                1.0e-6,
                msg=f"{label}: electrostatic system delta is zero"
            )

            # PotentialEnergy in upstream oxDNA is reported
            # per particle, whereas PairEnergy reports the total
            # interaction energy of the selected pair.
            #
            # Therefore:
            #
            #     d_pair = N * d_system
            #
            # for this micro-system, where all ES delta comes from
            # the selected pair.
            n_particles = int(
                TOP.splitlines()[0].split()[0]
            )

            expected_pair_delta = (
                n_particles * d_system
            )

            self.assertAlmostEqual(
                d_pair,
                expected_pair_delta,
                delta=max(
                    3.0e-6,
                    8.0e-4 * abs(expected_pair_delta),
                ),
                msg=(
                    f"{label}: PairEnergy does not expose "
                    f"the electrostatic term correctly after "
                    f"accounting for PotentialEnergy's "
                    f"per-particle normalisation: "
                    f"d_pair={d_pair:.12e}, "
                    f"N*d_system={expected_pair_delta:.12e}, "
                    f"N={n_particles}"
                ),
            )

    def test_dnanm_pd_pairenergy_contains_term12(self):
        self.delta_check(
            "PD",
            True,
            False,
            (1.0, 0.0, 0.0),
            (0, 1),
        )

    def test_dnanm_pp_pairenergy_contains_term13(self):
        self.delta_check(
            "PP",
            False,
            True,
            (1.0, 0.0, 1.0),
            (1, 3),
        )

    def test_dnanm_pairenergy_header_matches_registered_terms(self):
        with tempfile.TemporaryDirectory(
            prefix="oxdna_pairenergy_header_"
        ) as td:
            root = Path(td)

            run_case(
                root,
                "header",
                True,
                True,
                (1.0, 0.0, 1.0),
                (0, 1),
                print_header=True,
            )

            output = root / "header" / "pair_energy.dat"
            header = pairenergy_header(output)

            header_prefix = header.split(", t =", 1)[0]
            labels = header_prefix.split()

            expected = [
                "#id1",
                "id2",
                "FENE",
                "BEXC",
                "STCK",
                "NEXC",
                "HB",
                "CRSTCK",
                "CXSTCK",
                "DH",
                "SPRING",
                "PRO_EXC_VOL",
                "PRO_DNA_EXC_VOL",
                "PRO_DNA_DH",
                "PRO_PRO_DH",
                "total",
            ]

            self.assertEqual(labels, expected)

            numeric = first_numeric(output)

            self.assertEqual(
                len(numeric),
                len(expected),
                msg=(
                    "PairEnergy header/data column-count mismatch: "
                    f"header={len(expected)} numeric={len(numeric)}"
                ),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
