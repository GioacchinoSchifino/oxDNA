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
DT = 1.0e-4

DNA_REF = 0.815

DNA_POS = (
    5.0,
    5.0,
    5.0,
)

BACK = (
    -0.3400,
    +0.3408,
    0.0,
)

BACK_ABS = tuple(
    DNA_POS[i] + BACK[i]
    for i in range(3)
)

P1 = (
    BACK_ABS[0],
    BACK_ABS[1],
    BACK_ABS[2] + 1.02,
)

P2 = (
    12.0,
    12.0,
    12.0,
)

P3 = (
    P1[0] + 0.8,
    P1[1],
    P1[2],
)

POSITIONS = (
    DNA_POS,
    P1,
    P2,
    P3,
)

L_INIT = (
    1.0e-4,
    2.0e-4,
    3.0e-4,
)


TOP = """4 4 1 3 1
1 A -1 -1
-1 K -1 -1
-2 K -1 -1
-3 K -1 -1
"""


# ================================================================
# Utilities
# ================================================================

def vsub(a, b):
    return tuple(
        x - y
        for x, y in zip(a, b)
    )


def vnorm(a):
    return math.sqrt(
        sum(
            x * x
            for x in a
        )
    )


def distance(a, b):
    return vnorm(
        vsub(a, b)
    )


def zero_vec():
    return (
        0.0,
        0.0,
        0.0,
    )


def zero_tensor():
    return (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )


def add_vec(a, b):
    return tuple(
        x + y
        for x, y in zip(a, b)
    )


def sub_vec(a, b):
    return tuple(
        x - y
        for x, y in zip(a, b)
    )


def add_tensor(a, b):
    return tuple(
        x + y
        for x, y in zip(a, b)
    )


def sub_tensor(a, b):
    return tuple(
        x - y
        for x, y in zip(a, b)
    )


def max_vec_difference(a, b):
    return max(
        abs(x - y)
        for va, vb in zip(a, b)
        for x, y in zip(va, vb)
    )


def max_tensor_difference(a, b):
    return max(
        abs(x - y)
        for x, y in zip(a, b)
    )


def max_particle_tensor_difference(a, b):
    return max(
        abs(x - y)
        for ta, tb in zip(a, b)
        for x, y in zip(ta, tb)
    )


def assert_scalar_close(
    label,
    actual,
    expected,
    *,
    atol=2.0e-5,
    rtol=5.0e-3,
):
    if not (math.isfinite(actual) and math.isfinite(expected)):
        raise RuntimeError(f"{label}: nonfinite comparison actual={actual}, expected={expected}")
    if (
        abs(actual - expected)
        >
        atol
        +
        rtol * abs(expected)
    ):
        raise RuntimeError(
            f"{label}: actual={actual:.16e} "
            f"expected={expected:.16e}"
        )


def assert_vectors_close(
    label,
    actual,
    expected,
    *,
    atol=2.0e-5,
    rtol=5.0e-3,
):
    if len(actual) != len(expected) or any(len(v) != 3 for v in (*actual, *expected)):
        raise RuntimeError(f"{label}: vector shape mismatch")
    for i, (a, e) in enumerate(
        zip(actual, expected)
    ):
        for axis, (x, y) in enumerate(
            zip(a, e)
        ):
            assert_scalar_close(
                f"{label}/particle={i}/axis={axis}",
                x,
                y,
                atol=atol,
                rtol=rtol,
            )


def assert_tensor_close(
    label,
    actual,
    expected,
    *,
    atol=4.0e-6,
    rtol=5.0e-3,
):
    if len(actual) != 6 or len(expected) != 6:
        raise RuntimeError(f"{label}: tensor length mismatch")
    for i, (x, y) in enumerate(
        zip(actual, expected)
    ):
        assert_scalar_close(
            f"{label}/component={i}",
            x,
            y,
            atol=atol,
            rtol=rtol,
        )


# ================================================================
# Files
# ================================================================

def particle_line(pos):
    return (
        f"{pos[0]:.12f} "
        f"{pos[1]:.12f} "
        f"{pos[2]:.12f} "
        "1 0 0 "
        "0 0 1 "
        "0 0 0 "
        f"{L_INIT[0]:.12g} "
        f"{L_INIT[1]:.12g} "
        f"{L_INIT[2]:.12g}"
    )


def configuration():
    return (
        "t = 0\n"
        "b = 20 20 20\n"
        "E = 0 0 0\n"
        +
        "\n".join(
            particle_line(p)
            for p in POSITIONS
        )
        +
        "\n"
    )


def charge_file(charges):
    return f"""# DNANM_ES_CHARGES 1
# index_space = protein_local
# index_base = 0
# charge_model = integrated-cpu-cuda-parity
# charge_source = analytical
# n_protein_beads = 3
# n_groups = 3
# columns = protein_index topology_strand residue charge_e group_id
0 -1 K {charges[0]:.16g} 0
1 -2 K {charges[1]:.16g} 1
2 -3 K {charges[2]:.16g} 2
"""


def springs(mode):
    d12 = distance(P1, P2)
    d23 = distance(P2, P3)

    if mode == "DNACT":
        return (
            "3\n"
            f"1 2 {d12:.16g} "
            "s 0.0 "
            "0.0 0.0 0.0 0.0\n"
            f"2 3 {d23:.16g} "
            "s 0.0 "
            "0.0 0.0 0.0 0.0\n"
        )

    return (
        "3\n"
        f"1 2 {d12:.16g} s 0.0\n"
        f"2 3 {d23:.16g} s 0.0\n"
    )


def input_file(
    backend,
    mode,
    pd,
    pp,
):
    cuda = ""

    stress_cuda = ""

    if backend == "CUDA":
        cuda = """backend_precision = mixed
CUDA_list = verlet
CUDA_sort_every = 0
use_edge = 1
edge_n_forces = 1
"""

        stress_cuda = """CUDA_update_stress_tensor_every = 1
CUDA_update_particle_stress_tensor = true
"""

    angular = ""

    if mode == "DNACT":
        angular = (
            "bending_k = 0.0\n"
            "torsion_k = 0.0\n"
        )

    return f"""backend = {backend}
{cuda}
interaction_type = {mode}

T = 300K
salt_concentration = 1.0
dh_half_charged_ends = false

sim_type = MD

steps = 1
dt = {DT:.16g}

thermostat = no
refresh_vel = 0

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

print_conf_interval = 2
print_energy_every = 2

protein_dna_electrostatics = {"true" if pd else "false"}
protein_protein_electrostatics = {"true" if pp else "false"}

protein_charge_file = protein_charges.dat

protein_charge_scale = 1.0
protein_dna_dh_scale = 1.0
protein_protein_dh_scale = 1.0

dna_effective_charge_e = {DNA_REF:.16g}

exclude_intragroup_electrostatics = true

{stress_cuda}
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


# ================================================================
# Parsing
# ================================================================

def parse_state(path):
    rows = [
        x.strip()
        for x in path.read_text().splitlines()
        if x.strip()
    ]

    data = rows[3:3 + N]

    if len(data) != N:
        raise RuntimeError(
            f"Malformed state file {path}"
        )

    out = []

    for line in data:
        vals = tuple(
            float(x)
            for x in line.split()
        )

        if len(vals) != 15:
            raise RuntimeError(
                f"Expected 15 fields in {path}; "
                f"got {len(vals)}"
            )

        out.append(vals)

    return tuple(out)


def parse_pressure(path):
    numeric = []

    for line in path.read_text().splitlines():
        f = line.split()

        if len(f) != 7:
            continue

        try:
            row = tuple(float(x) for x in f)
        except ValueError:
            continue

        numeric.append(row)

    if not numeric:
        raise RuntimeError(
            f"No stress row in {path}"
        )

    # Drop scalar P.
    return numeric[-1][1:]


def parse_particle_stress(path):
    numeric = []

    for line in path.read_text().splitlines():
        f = line.split()

        if len(f) != 6:
            continue

        try:
            row = tuple(float(x) for x in f)
        except ValueError:
            continue

        numeric.append(row)

    if len(numeric) < N:
        raise RuntimeError(
            f"Expected >= {N} particle stress rows "
            f"in {path}; found {len(numeric)}"
        )

    return tuple(
        numeric[-N:]
    )


# ================================================================
# Execute
# ================================================================

def run(
    root,
    *,
    backend,
    mode,
    label,
    charges,
    pd,
    pp,
):
    case = (
        root
        / f"{backend}_{mode}_{label}"
    )

    case.mkdir()

    (
        case
        / "system.top"
    ).write_text(TOP)

    (
        case
        / "init.dat"
    ).write_text(configuration())

    (
        case
        / "protein_charges.dat"
    ).write_text(
        charge_file(charges)
    )

    (
        case
        / "springs.par"
    ).write_text(
        springs(mode)
    )

    (
        case
        / "input"
    ).write_text(
        input_file(
            backend,
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
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=90,
    )

    log_path = case / "log.dat"

    log = (
        log_path.read_text(
            errors="replace"
        )
        if log_path.exists()
        else ""
    )

    if proc.returncode != 0:
        raise RuntimeError(
            f"{case.name} failed rc="
            f"{proc.returncode}\n"
            f"{proc.stdout}\n{log}"
        )

    if (
        "END OF THE SIMULATION, everything went OK!"
        not in log
    ):
        raise RuntimeError(
            f"{case.name}: clean completion "
            "not confirmed"
        )

    if backend == "CUDA":
        if "Running on device 0" not in log:
            raise RuntimeError(
                f"{case.name}: CUDA device "
                "execution not confirmed"
            )

    return {
        "state": parse_state(
            case
            / "last_conf.dat"
        ),
        "stress": parse_pressure(
            case
            / "pressure.dat"
        ),
        "particle_stress": (
            parse_particle_stress(
                case
                / "particle_stress.dat"
            )
        ),
    }


# ================================================================
# ON-OFF response
# ================================================================

def response(off, on):
    forces = []
    torques = []

    for a, b in zip(
        off["state"],
        on["state"],
    ):

        forces.append(
            tuple(
                (
                    b[i]
                    - a[i]
                )
                / DT
                for i in range(
                    9,
                    12,
                )
            )
        )

        torques.append(
            tuple(
                (
                    b[i]
                    - a[i]
                )
                / DT
                for i in range(
                    12,
                    15,
                )
            )
        )

    stress = sub_tensor(
        on["stress"],
        off["stress"],
    )

    particle_stress = tuple(
        sub_tensor(a, b)
        for a, b in zip(
            on["particle_stress"],
            off["particle_stress"],
        )
    )

    return {
        "forces": tuple(forces),
        "torques": tuple(torques),
        "stress": stress,
        "particle_stress": particle_stress,
    }


# ================================================================
# Sanity
# ================================================================

def newton_error(forces):
    total = (
        sum(x[0] for x in forces),
        sum(x[1] for x in forces),
        sum(x[2] for x in forces),
    )

    return vnorm(total)


def particle_sum(particle_stress):
    total = zero_tensor()

    for t in particle_stress:
        total = add_tensor(
            total,
            t,
        )

    return total


def response_norm(response):
    return max(
        abs(x)
        for v in response["forces"]
        for x in v
    )


# ================================================================
# Cases
# ================================================================

SPECS = {
    "PD": {
        "charges": (
            +1.0,
            0.0,
            0.0,
        ),
        "pd": True,
        "pp": False,
    },

    # Same charge layout as BOTH but PD only.
    # This allows an exact superposition test.
    "PD2": {
        "charges": (
            +1.0,
            0.0,
            +1.0,
        ),
        "pd": True,
        "pp": False,
    },

    "PP": {
        "charges": (
            +1.0,
            0.0,
            +1.0,
        ),
        "pd": False,
        "pp": True,
    },

    "BOTH": {
        "charges": (
            +1.0,
            0.0,
            +1.0,
        ),
        "pd": True,
        "pp": True,
    },
}


def main():

    results = {}

    with tempfile.TemporaryDirectory(
        prefix="oxdna_cpu_cuda_parity29_"
    ) as td:

        root = Path(td)

        for backend in (
            "CPU",
            "CUDA",
        ):
            for mode in (
                "DNANM",
                "DNACT",
            ):
                for label, spec in SPECS.items():

                    off = run(
                        root,
                        backend=backend,
                        mode=mode,
                        label=f"{label}_OFF",
                        charges=spec["charges"],
                        pd=False,
                        pp=False,
                    )

                    on = run(
                        root,
                        backend=backend,
                        mode=mode,
                        label=f"{label}_ON",
                        charges=spec["charges"],
                        pd=spec["pd"],
                        pp=spec["pp"],
                    )

                    r = response(
                        off,
                        on,
                    )

                    results[
                        (
                            backend,
                            mode,
                            label,
                        )
                    ] = r

                    nerr = newton_error(
                        r["forces"]
                    )

                    print(
                        "INTEGRATED_CASE",
                        backend,
                        mode,
                        label,
                        "maxF=",
                        f"{response_norm(r):.16e}",
                        "newton=",
                        f"{nerr:.16e}",
                        "stress=",
                        r["stress"],
                    )

                    if label != "BOTH" and label != "PD2":
                        if (
                            response_norm(r)
                            < 1.0e-7
                        ):
                            raise RuntimeError(
                                f"{backend}/{mode}/{label}: "
                                "electrostatic force response "
                                "unexpectedly zero"
                            )

                    if nerr > 3.0e-5:
                        raise RuntimeError(
                            f"{backend}/{mode}/{label}: "
                            f"Newton III failure {nerr}"
                        )

                    # Particle stress must sum to total stress.
                    psum = particle_sum(
                        r["particle_stress"]
                    )

                    assert_tensor_close(
                        f"{backend}/{mode}/{label}/"
                        "particle_sum",
                        psum,
                        r["stress"],
                        atol=7.0e-6,
                        rtol=8.0e-3,
                    )

    # ============================================================
    # CPU <-> CUDA direct parity
    # ============================================================

    for mode in (
        "DNANM",
        "DNACT",
    ):
        for label in SPECS:

            cpu = results[
                (
                    "CPU",
                    mode,
                    label,
                )
            ]

            gpu = results[
                (
                    "CUDA",
                    mode,
                    label,
                )
            ]

            assert_vectors_close(
                f"{mode}/{label}/CPU_CUDA_FORCE",
                gpu["forces"],
                cpu["forces"],
                atol=2.5e-5,
                rtol=6.0e-3,
            )

            assert_vectors_close(
                f"{mode}/{label}/CPU_CUDA_TORQUE",
                gpu["torques"],
                cpu["torques"],
                atol=2.5e-5,
                rtol=8.0e-3,
            )

            assert_tensor_close(
                f"{mode}/{label}/CPU_CUDA_STRESS",
                gpu["stress"],
                cpu["stress"],
                atol=8.0e-6,
                rtol=8.0e-3,
            )

            for i in range(N):
                assert_tensor_close(
                    f"{mode}/{label}/CPU_CUDA_"
                    f"PARTICLE_STRESS/{i}",
                    gpu["particle_stress"][i],
                    cpu["particle_stress"][i],
                    atol=8.0e-6,
                    rtol=8.0e-3,
                )

            fd = max_vec_difference(
                gpu["forces"],
                cpu["forces"],
            )

            td = max_vec_difference(
                gpu["torques"],
                cpu["torques"],
            )

            sd = max_tensor_difference(
                gpu["stress"],
                cpu["stress"],
            )

            psd = max_particle_tensor_difference(
                gpu["particle_stress"],
                cpu["particle_stress"],
            )

            print(
                "CPU_CUDA_PARITY",
                mode,
                label,
                f"force_max_delta={fd:.16e}",
                f"torque_max_delta={td:.16e}",
                f"stress_max_delta={sd:.16e}",
                f"particle_stress_max_delta={psd:.16e}",
            )

    # ============================================================
    # DNANM <-> DNACT parity within each backend
    # ============================================================

    for backend in (
        "CPU",
        "CUDA",
    ):
        for label in SPECS:

            a = results[
                (
                    backend,
                    "DNANM",
                    label,
                )
            ]

            b = results[
                (
                    backend,
                    "DNACT",
                    label,
                )
            ]

            assert_vectors_close(
                f"{backend}/{label}/DNANM_DNACT_FORCE",
                a["forces"],
                b["forces"],
                atol=2.0e-5,
                rtol=5.0e-3,
            )

            assert_vectors_close(
                f"{backend}/{label}/DNANM_DNACT_TORQUE",
                a["torques"],
                b["torques"],
                atol=2.0e-5,
                rtol=5.0e-3,
            )

            assert_tensor_close(
                f"{backend}/{label}/DNANM_DNACT_STRESS",
                a["stress"],
                b["stress"],
                atol=6.0e-6,
                rtol=6.0e-3,
            )

    # ============================================================
    # Additivity:
    #
    # BOTH(q1,q3) == PD2(q1,q3) + PP(q1,q3)
    #
    # independently on CPU and CUDA.
    # ============================================================

    for backend in (
        "CPU",
        "CUDA",
    ):
        for mode in (
            "DNANM",
            "DNACT",
        ):

            pd2 = results[
                (
                    backend,
                    mode,
                    "PD2",
                )
            ]

            pp = results[
                (
                    backend,
                    mode,
                    "PP",
                )
            ]

            both = results[
                (
                    backend,
                    mode,
                    "BOTH",
                )
            ]

            expected_f = tuple(
                add_vec(a, b)
                for a, b in zip(
                    pd2["forces"],
                    pp["forces"],
                )
            )

            expected_t = tuple(
                add_vec(a, b)
                for a, b in zip(
                    pd2["torques"],
                    pp["torques"],
                )
            )

            expected_s = add_tensor(
                pd2["stress"],
                pp["stress"],
            )

            expected_ps = tuple(
                add_tensor(a, b)
                for a, b in zip(
                    pd2["particle_stress"],
                    pp["particle_stress"],
                )
            )

            assert_vectors_close(
                f"{backend}/{mode}/BOTH_ADDITIVITY_FORCE",
                both["forces"],
                expected_f,
                atol=3.0e-5,
                rtol=7.0e-3,
            )

            assert_vectors_close(
                f"{backend}/{mode}/BOTH_ADDITIVITY_TORQUE",
                both["torques"],
                expected_t,
                atol=3.0e-5,
                rtol=8.0e-3,
            )

            assert_tensor_close(
                f"{backend}/{mode}/BOTH_ADDITIVITY_STRESS",
                both["stress"],
                expected_s,
                atol=8.0e-6,
                rtol=8.0e-3,
            )

            for i in range(N):
                assert_tensor_close(
                    f"{backend}/{mode}/BOTH_ADDITIVITY_"
                    f"PARTICLE_STRESS/{i}",
                    both["particle_stress"][i],
                    expected_ps[i],
                    atol=8.0e-6,
                    rtol=8.0e-3,
                )

            print(
                "PD_PP_ADDITIVITY",
                backend,
                mode,
                "PASS",
            )

    print("CPU_CUDA_PD_FORCE_PARITY=PASS")
    print("CPU_CUDA_PP_FORCE_PARITY=PASS")
    print("CPU_CUDA_BOTH_FORCE_PARITY=PASS")

    print("CPU_CUDA_PD_TORQUE_PARITY=PASS")
    print("CPU_CUDA_PP_TORQUE_PARITY=PASS")
    print("CPU_CUDA_BOTH_TORQUE_PARITY=PASS")

    print("CPU_CUDA_PD_STRESS_PARITY=PASS")
    print("CPU_CUDA_PP_STRESS_PARITY=PASS")
    print("CPU_CUDA_BOTH_STRESS_PARITY=PASS")

    print("CPU_CUDA_PARTICLE_STRESS_PARITY=PASS")
    print("CPU_CUDA_PARTICLE_SUM_TOTAL_PARITY=PASS")

    print("CPU_DNANM_DNACT_PARITY=PASS")
    print("CUDA_DNANM_DNACT_INTEGRATED_PARITY=PASS")

    print("CPU_PD_PP_ADDITIVITY=PASS")
    print("CUDA_PD_PP_ADDITIVITY=PASS")

    print("INTEGRATED_CPU_CUDA_PARITY_GATE=PASS")


if __name__ == "__main__":
    main()
