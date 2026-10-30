#!/usr/bin/env python3

import importlib.util
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve()
CORE_PATH = HERE.with_name(
    "test_analytical_physics_core.py"
)

spec = importlib.util.spec_from_file_location(
    "analytical_core",
    CORE_PATH,
)

A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)

OXDNA = Path(
    os.environ["OXDNA_BIN"]
).resolve()


def tensor6(r, f):
    """
    oxDNA StressTensor ordering:
      xx yy zz xy xz yz

    W_ab = r_a F_b
    """
    return (
        r[0] * f[0],
        r[1] * f[1],
        r[2] * f[2],
        r[0] * f[1],
        r[0] * f[2],
        r[1] * f[2],
    )


def vec_sub(a, b):
    return tuple(
        x - y
        for x, y in zip(a, b)
    )


def vec_scale(a, s):
    return tuple(
        s * x
        for x in a
    )


def input_text(
    interaction_type,
    pd,
    pp,
):
    angular = ""
    parfile = "none"

    if interaction_type == "DNACT":
        parfile = "empty_network.par"
        angular = (
            "bending_k = 0.0\n"
            "torsion_k = 0.0\n"
        )

    return f"""backend = CPU
interaction_type = {interaction_type}

sim_type = MD

T = 300K
dt = 0.002

thermostat = no
refresh_vel = 1
seed = 314159

restart_step_counter = 1
time_scale = linear

verlet_skin = 0.25

ensemble = NVT
external_forces = 0

max_io = 100

salt_concentration = 1.0
dh_half_charged_ends = false

{angular}parfile = {parfile}

topology = system.top
conf_file = init.dat

trajectory_file = trajectory.dat
lastconf_file = last_conf.dat
energy_file = energy.dat
log_file = log.dat

no_stdout_energy = 1

steps = 1
print_conf_interval = 1
print_energy_every = 1

protein_dna_electrostatics = {"true" if pd else "false"}
protein_protein_electrostatics = {"true" if pp else "false"}

protein_charge_file = protein_charges.dat

protein_charge_scale = 1.0
protein_dna_dh_scale = 1.0
protein_protein_dh_scale = 1.0

dna_effective_charge_e = 0.815

exclude_intragroup_electrostatics = true

data_output_1 = {{
 name = pressure.dat
 print_every = 2

 col_1 = {{
  type = pressure
  custom_stress_tensor = true
  stress_tensor = true
  PV_only = true
 }}
}}

data_output_2 = {{
 name = particle_stress.dat
 print_every = 2

 col_1 = {{
  type = particle_stress
  print_coordinates = false
 }}
}}
"""


def numeric_rows(path):
    rows = []

    for raw in path.read_text().splitlines():

        line = raw.strip()

        if not line:
            continue

        if line.startswith("#"):
            continue

        try:
            row = [
                float(x)
                for x in line.split()
            ]
        except ValueError:
            continue

        rows.append(row)

    return rows


def run_case(
    root,
    name,
    *,
    interaction_type,
    pd,
    pp,
    charges,
):
    case = root / name
    case.mkdir()

    (case / "system.top").write_text(
        A.TOP
    )

    if interaction_type == "DNACT":
        # Keep the former no-spring fixture mechanics with an explicit file.
        n_protein = int(A.TOP.splitlines()[0].split()[3])
        (case / "empty_network.par").write_text(f"{n_protein}\n")

    (case / "init.dat").write_text(
        A.conf_text()
    )

    (case / "protein_charges.dat").write_text(
        A.charges_text(
            charges,
            (0, 1, 2),
        )
    )

    (case / "input").write_text(
        input_text(
            interaction_type,
            pd,
            pp,
        )
    )

    proc = subprocess.run(
        [
            str(OXDNA),
            "input",
        ],
        cwd=case,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
    )

    log = ""

    if (case / "log.dat").exists():
        log = (
            case / "log.dat"
        ).read_text(
            errors="replace"
        )

    if proc.returncode != 0:
        raise RuntimeError(
            f"{name} failed\n"
            f"{proc.stdout}\n"
            f"{log}"
        )

    pressure_rows = numeric_rows(
        case / "pressure.dat"
    )

    if not pressure_rows:
        raise RuntimeError(
            f"No Pressure rows in {name}"
        )

    # Pressure with:
    #   stress_tensor = true
    #   PV_only = true
    #
    # outputs:
    #   P xx yy zz xy xz yz
    #
    # where the six stress components have been multiplied
    # back by V, i.e. raw virial/stress sums.
    pressure = pressure_rows[0]

    if len(pressure) != 7:
        raise RuntimeError(
            f"Unexpected Pressure row in {name}: "
            f"{pressure}"
        )

    total_raw = tuple(
        pressure[1:7]
    )

    particle_rows = numeric_rows(
        case / "particle_stress.dat"
    )

    # There are 4 particles.  We only need the first output block.
    if len(particle_rows) < 4:
        raise RuntimeError(
            f"Not enough ParticleStress rows in {name}: "
            f"{particle_rows}"
        )

    particle_raw = []

    for row in particle_rows[:4]:

        if len(row) != 6:
            raise RuntimeError(
                f"Unexpected ParticleStress row "
                f"in {name}: {row}"
            )

        particle_raw.append(
            tuple(row)
        )

    return {
        "total": total_raw,
        "particle": particle_raw,
    }


def subtract(a, b):
    return tuple(
        x - y
        for x, y in zip(a, b)
    )


class StressTensorContract(unittest.TestCase):

    def assert_tensor_close(
        self,
        observed,
        expected,
        *,
        label,
        atol=3.0e-7,
        rtol=2.0e-5,
    ):
        for i, (obs, exp) in enumerate(
            zip(
                observed,
                expected,
            )
        ):
            tol = max(
                atol,
                rtol * abs(exp),
            )

            self.assertAlmostEqual(
                obs,
                exp,
                delta=tol,
                msg=(
                    f"{label}[{i}]: "
                    f"observed={obs:.12e}, "
                    f"expected={exp:.12e}, "
                    f"tol={tol:.3e}"
                ),
            )

    def run_pd_contract(
        self,
        interaction_type,
    ):
        with tempfile.TemporaryDirectory(
            prefix="oxdna_stress_pd_"
        ) as td:

            root = Path(td)

            off = run_case(
                root,
                "off",
                interaction_type=interaction_type,
                pd=False,
                pp=False,
                charges=(
                    +1.0,
                    0.0,
                    0.0,
                ),
            )

            on = run_case(
                root,
                "on",
                interaction_type=interaction_type,
                pd=True,
                pp=False,
                charges=(
                    +1.0,
                    0.0,
                    0.0,
                ),
            )

            total_delta = subtract(
                on["total"],
                off["total"],
            )

            particle_delta = [
                subtract(
                    on["particle"][i],
                    off["particle"][i],
                )
                for i in range(4)
            ]

            # BACK -> protein vector controls electrostatic force.
            rback = (
                0.0,
                0.0,
                1.02,
            )

            charge_product = (
                -1.0
                /
                A.DNA_EFFECTIVE_CHARGE_E
            )

            _, force_on_protein = (
                A.dh_energy_force_on_second(
                    charge_product,
                    rback,
                )
            )

            # Stress implementation deliberately uses COM separation,
            # not BACK-to-protein separation.
            rcenter = vec_sub(
                A.P1,
                A.DNA_POS,
            )

            expected = tensor6(
                rcenter,
                force_on_protein,
            )

            expected_half = vec_scale(
                expected,
                0.5,
            )

            zero = (
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )

            self.assert_tensor_close(
                total_delta,
                expected,
                label=f"{interaction_type} PD total",
            )

            # DNA particle 0 and protein particle 1 receive 1/2 each.
            self.assert_tensor_close(
                particle_delta[0],
                expected_half,
                label=f"{interaction_type} PD DNA half",
            )

            self.assert_tensor_close(
                particle_delta[1],
                expected_half,
                label=f"{interaction_type} PD protein half",
            )

            self.assert_tensor_close(
                particle_delta[2],
                zero,
                label=f"{interaction_type} PD spectator 2",
            )

            self.assert_tensor_close(
                particle_delta[3],
                zero,
                label=f"{interaction_type} PD spectator 3",
            )

            summed = tuple(
                sum(
                    particle_delta[p][k]
                    for p in range(4)
                )
                for k in range(6)
            )

            self.assert_tensor_close(
                summed,
                total_delta,
                label=f"{interaction_type} PD sum(particle)=total",
            )

            print(
                "PD_STRESS",
                interaction_type,
                "observed=",
                total_delta,
                "expected=",
                expected,
            )

    def run_pp_contract(
        self,
        interaction_type,
    ):
        with tempfile.TemporaryDirectory(
            prefix="oxdna_stress_pp_"
        ) as td:

            root = Path(td)

            off = run_case(
                root,
                "off",
                interaction_type=interaction_type,
                pd=False,
                pp=False,
                charges=(
                    +1.0,
                    0.0,
                    +1.0,
                ),
            )

            on = run_case(
                root,
                "on",
                interaction_type=interaction_type,
                pd=False,
                pp=True,
                charges=(
                    +1.0,
                    0.0,
                    +1.0,
                ),
            )

            total_delta = subtract(
                on["total"],
                off["total"],
            )

            particle_delta = [
                subtract(
                    on["particle"][i],
                    off["particle"][i],
                )
                for i in range(4)
            ]

            rcenter = vec_sub(
                A.P3,
                A.P1,
            )

            qred = (
                1.0
                /
                A.DNA_EFFECTIVE_CHARGE_E
            )

            _, force_on_p3 = (
                A.dh_energy_force_on_second(
                    qred * qred,
                    rcenter,
                )
            )

            expected = tensor6(
                rcenter,
                force_on_p3,
            )

            expected_half = vec_scale(
                expected,
                0.5,
            )

            zero = (
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )

            self.assert_tensor_close(
                total_delta,
                expected,
                label=f"{interaction_type} PP total",
            )

            # Protein particles are global IDs 1 and 3.
            self.assert_tensor_close(
                particle_delta[1],
                expected_half,
                label=f"{interaction_type} PP first half",
            )

            self.assert_tensor_close(
                particle_delta[3],
                expected_half,
                label=f"{interaction_type} PP second half",
            )

            self.assert_tensor_close(
                particle_delta[0],
                zero,
                label=f"{interaction_type} PP DNA spectator",
            )

            self.assert_tensor_close(
                particle_delta[2],
                zero,
                label=f"{interaction_type} PP protein spectator",
            )

            summed = tuple(
                sum(
                    particle_delta[p][k]
                    for p in range(4)
                )
                for k in range(6)
            )

            self.assert_tensor_close(
                summed,
                total_delta,
                label=f"{interaction_type} PP sum(particle)=total",
            )

            print(
                "PP_STRESS",
                interaction_type,
                "observed=",
                total_delta,
                "expected=",
                expected,
            )

    def test_01_dnanm_pd_total_and_particle_stress(self):
        self.run_pd_contract(
            "DNANM"
        )

    def test_02_dnanm_pp_total_and_particle_stress(self):
        self.run_pp_contract(
            "DNANM"
        )

    def test_03_dnact_pd_total_and_particle_stress(self):
        self.run_pd_contract(
            "DNACT"
        )

    def test_04_dnact_pp_total_and_particle_stress(self):
        self.run_pp_contract(
            "DNACT"
        )


if __name__ == "__main__":
    unittest.main(
        verbosity=2
    )
