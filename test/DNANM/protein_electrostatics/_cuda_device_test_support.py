"""Shared CUDA runtime-test helpers for DNANM protein electrostatics.

This module contains deterministic system builders, parsers and
validation helpers used by the CUDA device regression tests.
"""

#!/usr/bin/env python3

import math
import os
import subprocess
import tempfile
from pathlib import Path


OXDNA = Path(
    os.environ["OXDNA_BIN"]
).resolve()

if not OXDNA.is_file():
    raise RuntimeError(
        f"Invalid OXDNA_BIN: {OXDNA}"
    )


N = 4

DNA_EFFECTIVE_CHARGE_E = 0.815
DH_STRENGTH = 0.0543
DH_LAMBDA = 0.3616455

DNA_POS = (
    5.0,
    5.0,
    5.0,
)

P1 = (
    4.6600,
    5.3408,
    6.0200,
)

P2 = (
    12.0,
    12.0,
    12.0,
)

P3 = (
    5.4600,
    5.3408,
    6.0200,
)


TOP = """4 4 1 3 1
1 A -1 -1
-1 K -1 -1
-2 K -1 -1
-3 K -1 -1
"""


def particle_line(pos):
    return (
        f"{pos[0]:.8f} "
        f"{pos[1]:.8f} "
        f"{pos[2]:.8f} "
        "1 0 0 "
        "0 0 1 "
        "0 0 0 "
        "0 0 0"
    )


CONF = (
    "t = 0\n"
    "b = 20 20 20\n"
    "E = 0 0 0\n"
    + particle_line(DNA_POS)
    + "\n"
    + particle_line(P1)
    + "\n"
    + particle_line(P2)
    + "\n"
    + particle_line(P3)
    + "\n"
)


def charges_text(charges):
    return """# DNANM_ES_CHARGES 1
# index_space = protein_local
# index_base = 0
# charge_model = cuda-device-test
# charge_source = analytical
# n_protein_beads = 3
# n_groups = 3
# columns = protein_index topology_strand residue charge_e group_id
""" + (
        f"0 -1 K {charges[0]:.16g} 0\n"
        f"1 -2 K {charges[1]:.16g} 1\n"
        f"2 -3 K {charges[2]:.16g} 2\n"
    )


def springs_text(mode):
    # These zero-k springs exist only because current upstream
    # CUDA DNANM expects allocated ANM connectivity arrays.
    #
    # Global protein indices are 1,2,3 because global 0 is DNA.
    #
    # Tested PP pair = 1--3, which is deliberately NOT connected.
    if mode == "DNACT":
        return """3
1 2 11.575013807335178 s 0.0 0.0 0.0 0.0 0.0
2 3 11.084987354074880 s 0.0 0.0 0.0 0.0 0.0
"""

    return """3
1 2 11.575013807335178 s 0.0
2 3 11.084987354074880 s 0.0
"""


def input_text(
    mode,
    pd,
    pp,
):
    angular = ""

    if mode == "DNACT":
        angular = (
            "bending_k = 0.0\n"
            "torsion_k = 0.0\n"
        )

    return f"""backend = CUDA
backend_precision = mixed

CUDA_list = verlet
CUDA_sort_every = 0

use_edge = 1
edge_n_forces = 1

interaction_type = {mode}

T = 300K
salt_concentration = 1.0
dh_half_charged_ends = false

sim_type = MD

# Twenty deterministic MD steps amplify any real device-side
# electrostatic force enough to be visible in the final state.
steps = 20
dt = 0.002

thermostat = no
refresh_vel = 1
seed = 271828

restart_step_counter = 1
time_scale = linear

verlet_skin = 0.25
max_density_multiplier = 100

{angular}parfile = springs.par

topology = system.top
conf_file = init.dat

trajectory_file = trajectory.dat
lastconf_file = last_conf.dat

energy_file = energy.dat
log_file = log.dat

no_stdout_energy = 1

print_conf_interval = 20
print_energy_every = 20

protein_dna_electrostatics = {"true" if pd else "false"}
protein_protein_electrostatics = {"true" if pp else "false"}

protein_charge_file = protein_charges.dat

protein_charge_scale = 1.0
protein_dna_dh_scale = 1.0
protein_protein_dh_scale = 1.0

dna_effective_charge_e = 0.815

exclude_intragroup_electrostatics = true
"""


def first_potential(path):
    for raw in path.read_text().splitlines():

        fields = raw.split()

        if len(fields) < 2:
            continue

        try:
            vals = [
                float(x)
                for x in fields
            ]
        except ValueError:
            continue

        return vals[1]

    raise RuntimeError(
        f"No numeric energy row in {path}"
    )


def particle_state(path):
    lines = [
        x.strip()
        for x in path.read_text().splitlines()
        if x.strip()
    ]

    if len(lines) < 3 + N:
        raise RuntimeError(
            f"Malformed configuration: {path}\n"
            + "\n".join(lines)
        )

    if not lines[0].startswith("t = "):
        raise RuntimeError(
            f"Missing t header in {path}"
        )

    if not lines[1].startswith("b = "):
        raise RuntimeError(
            f"Missing b header in {path}"
        )

    if not lines[2].startswith("E = "):
        raise RuntimeError(
            f"Missing E header in {path}"
        )

    # IMPORTANT:
    # Ignore the first three configuration headers.
    # In particular E= contains host-side TotalEnergy.
    #
    # What follows is the actual particle dynamical state:
    # position, orientation, velocity, angular momentum.
    rows = tuple(
        lines[
            3:
            3 + N
        ]
    )

    if len(rows) != N:
        raise RuntimeError(
            f"Expected {N} particle rows"
        )

    return rows


def numeric_state(rows):
    return tuple(
        tuple(
            float(x)
            for x in row.split()
        )
        for row in rows
    )


def max_state_delta(a, b):
    aa = numeric_state(a)
    bb = numeric_state(b)

    return max(
        abs(x - y)
        for ra, rb in zip(aa, bb)
        for x, y in zip(ra, rb)
    )


def run_case(
    root,
    name,
    *,
    mode,
    pd,
    pp,
    charges,
):
    case = root / name
    case.mkdir()

    (case / "system.top").write_text(
        TOP
    )

    (case / "init.dat").write_text(
        CONF
    )

    (case / "protein_charges.dat").write_text(
        charges_text(charges)
    )

    (case / "springs.par").write_text(
        springs_text(mode)
    )

    (case / "input").write_text(
        input_text(
            mode,
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
            f"{name} failed rc={proc.returncode}\n\n"
            f"{proc.stdout}\n\n"
            f"{log}"
        )

    if "Running on device 0" not in log:
        raise RuntimeError(
            f"{name}: CUDA execution not confirmed\n"
            f"{log}"
        )

    last_conf = case / "last_conf.dat"

    if not last_conf.exists():
        raise RuntimeError(
            f"{name}: last_conf.dat missing"
        )

    energy = case / "energy.dat"

    if not energy.exists():
        raise RuntimeError(
            f"{name}: energy.dat missing"
        )

    return {
        "potential": first_potential(
            energy
        ),
        "state": particle_state(
            last_conf
        ),
    }


def dh_direct(
    charge_product,
    r,
):
    return (
        DH_STRENGTH
        * charge_product
        * math.exp(
            -r / DH_LAMBDA
        )
        / r
    )


def require_host_energy(
    label,
    off,
    on,
    expected_system,
):
    observed = (
        on["potential"]
        - off["potential"]
    )

    expected_per_particle = (
        expected_system / N
    )

    tol = max(
        3.0e-6,
        8.0e-4
        * abs(expected_per_particle),
    )

    print(
        "HOST_ENERGY_CONTROL",
        label,
        f"delta={observed:.12e}",
        f"expected_per_particle={expected_per_particle:.12e}",
    )

    if abs(
        observed
        - expected_per_particle
    ) > tol:
        raise RuntimeError(
            f"{label}: host energy control failed: "
            f"observed={observed}, "
            f"expected/N={expected_per_particle}"
        )


def require_deterministic_off(
    label,
    a,
    b,
):
    delta = max_state_delta(
        a["state"],
        b["state"],
    )

    print(
        "CUDA_OFF_REPRODUCIBILITY",
        label,
        f"max_state_delta={delta:.16e}",
    )

    if a["state"] != b["state"]:
        raise RuntimeError(
            f"{label}: duplicate OFF CUDA runs are "
            f"not bitwise-identical at configuration precision; "
            f"max delta={delta}"
        )
