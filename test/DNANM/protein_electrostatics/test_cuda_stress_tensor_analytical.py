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


# ================================================================
# Model constants
# ================================================================

N = 4

DNA_EFFECTIVE_CHARGE_E = 0.815

DH_PREF = 0.0543
DH_LAMBDA = 0.3616455

DH_RHIGH = (
    3.0
    * DH_LAMBDA
)

DH_B = -(
    math.exp(
        -DH_RHIGH
        / DH_LAMBDA
    )
    *
    DH_PREF
    *
    DH_PREF
    *
    (
        DH_RHIGH
        + DH_LAMBDA
    ) ** 2
) / (
    -4.0
    *
    DH_RHIGH**3
    *
    DH_LAMBDA**2
    *
    DH_PREF
)

DH_RC = (
    DH_RHIGH
    *
    (
        DH_PREF
        * DH_RHIGH
        +
        3.0
        * DH_PREF
        * DH_LAMBDA
    )
    /
    (
        DH_PREF
        *
        (
            DH_RHIGH
            + DH_LAMBDA
        )
    )
)


# Identity-orientation BACK site.
BACK = (
    -0.3400,
    +0.3408,
    0.0,
)

DNA_POS = (
    5.0,
    5.0,
    5.0,
)

BACK_ABS = tuple(
    DNA_POS[i]
    + BACK[i]
    for i in range(3)
)

# Non-zero angular momentum only to keep the subsequent
# one-step CUDA orientation update well-defined.
#
# Stress is sampled BEFORE that MD step, so this does not
# contribute to translational kinetic stress.
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
# Vector / tensor utilities
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


def scale_tensor(t, k):
    return tuple(
        k * x
        for x in t
    )


def add_tensors(a, b):
    return tuple(
        x + y
        for x, y in zip(a, b)
    )


def sub_tensors(a, b):
    return tuple(
        x - y
        for x, y in zip(a, b)
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


def outer_stress(
    r,
    force,
):
    # Exact BaseInteraction tensor order:
    #
    # xx yy zz xy xz yz
    return (
        r[0] * force[0],
        r[1] * force[1],
        r[2] * force[2],
        r[0] * force[1],
        r[0] * force[2],
        r[1] * force[2],
    )


# ================================================================
# Analytical Debye-Huckel force
# ================================================================

def dh_force_on_second(
    rvec,
    charge_product,
):
    r = vnorm(rvec)

    if r <= 0.0:
        raise RuntimeError(
            "Invalid zero DH distance"
        )

    if r >= DH_RC:
        return (
            0.0,
            0.0,
            0.0,
        )

    rhat = tuple(
        x / r
        for x in rvec
    )

    if r < DH_RHIGH:

        exponential = math.exp(
            -r
            / DH_LAMBDA
        )

        scalar = (
            DH_PREF
            *
            exponential
            *
            (
                1.0
                /
                (
                    DH_LAMBDA
                    * r
                )
                +
                1.0
                /
                (
                    r
                    * r
                )
            )
            *
            charge_product
        )

    else:

        scalar = (
            -2.0
            *
            DH_B
            *
            (
                r
                - DH_RC
            )
            *
            charge_product
        )

    return tuple(
        scalar * x
        for x in rhat
    )


# ================================================================
# Geometry
# ================================================================

def positions():
    # P1:
    # exactly 1.02 from nucleotide BACK site.
    p1 = (
        BACK_ABS[0],
        BACK_ABS[1],
        BACK_ABS[2]
        + 1.02,
    )

    p2 = (
        12.0,
        12.0,
        12.0,
    )

    # P3:
    # exactly 0.8 from P1 along +x.
    p3 = (
        p1[0] + 0.8,
        p1[1],
        p1[2],
    )

    return (
        DNA_POS,
        p1,
        p2,
        p3,
    )


POSITIONS = positions()


def particle_line(pos):
    # velocity = exactly zero
    #
    # orientation:
    # v1 = (1,0,0)
    # v3 = (0,0,1)
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


def conf_text():
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


# ================================================================
# Charge / spring files
# ================================================================

def charges_text(
    charges,
    groups,
):
    return f"""# DNANM_ES_CHARGES 1
# index_space = protein_local
# index_base = 0
# charge_model = cuda-stress-analytical
# charge_source = analytical
# n_protein_beads = 3
# n_groups = {len(set(groups))}
# columns = protein_index topology_strand residue charge_e group_id
0 -1 K {charges[0]:.16g} {groups[0]}
1 -2 K {charges[1]:.16g} {groups[1]}
2 -3 K {charges[2]:.16g} {groups[2]}
"""


def springs_text(
    mode,
    connect_13=False,
):
    p1 = POSITIONS[1]
    p2 = POSITIONS[2]
    p3 = POSITIONS[3]

    d12 = distance(
        p1,
        p2,
    )

    d23 = distance(
        p2,
        p3,
    )

    d13 = distance(
        p1,
        p3,
    )

    lines = [
        "3",
    ]

    if mode == "DNACT":
        lines.append(
            f"1 2 {d12:.16g} "
            "s 0.0 "
            "0.0 0.0 0.0 0.0"
        )

        lines.append(
            f"2 3 {d23:.16g} "
            "s 0.0 "
            "0.0 0.0 0.0 0.0"
        )
    else:
        lines.append(
            f"1 2 {d12:.16g} s 0.0"
        )

        lines.append(
            f"2 3 {d23:.16g} s 0.0"
        )

    if connect_13:
        lines.append(
            f"1 3 {d13:.16g} s 0.0"
        )

    return (
        "\n".join(lines)
        + "\n"
    )


# ================================================================
# Simulation input
# ================================================================

def input_text(
    mode,
    *,
    pd,
    pp,
    exclude_group,
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

steps = 1
dt = 1e-4

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

dna_effective_charge_e = {DNA_EFFECTIVE_CHARGE_E:.16g}

exclude_intragroup_electrostatics = {"true" if exclude_group else "false"}

CUDA_update_stress_tensor_every = 1
CUDA_update_particle_stress_tensor = true

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
# Runtime
# ================================================================

def parse_pressure(path):
    rows = []

    for line in path.read_text().splitlines():
        fields = line.split()

        if len(fields) != 7:
            continue

        try:
            vals = tuple(
                float(x)
                for x in fields
            )
        except ValueError:
            continue

        rows.append(vals)

    if not rows:
        raise RuntimeError(
            f"No pressure/stress rows in {path}"
        )

    # P, xx, yy, zz, xy, xz, yz.
    return rows[-1][1:]


def parse_particle_stress(path):
    rows = []

    for line in path.read_text().splitlines():
        fields = line.split()

        if len(fields) != 6:
            continue

        try:
            vals = tuple(
                float(x)
                for x in fields
            )
        except ValueError:
            continue

        rows.append(vals)

    if len(rows) < N:
        raise RuntimeError(
            f"Expected at least {N} particle "
            f"stress rows in {path}; got {len(rows)}"
        )

    # Only the last output block, robust even if more than one
    # output is ever emitted.
    return tuple(
        rows[-N:]
    )


def run_case(
    root,
    name,
    *,
    mode,
    pd,
    pp,
    charges,
    groups,
    exclude_group=True,
    connect_13=False,
):
    case = root / name
    case.mkdir()

    (
        case
        / "system.top"
    ).write_text(
        TOP
    )

    (
        case
        / "init.dat"
    ).write_text(
        conf_text()
    )

    (
        case
        / "protein_charges.dat"
    ).write_text(
        charges_text(
            charges,
            groups,
        )
    )

    (
        case
        / "springs.par"
    ).write_text(
        springs_text(
            mode,
            connect_13,
        )
    )

    (
        case
        / "input"
    ).write_text(
        input_text(
            mode,
            pd=pd,
            pp=pp,
            exclude_group=exclude_group,
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
        timeout=60,
    )

    log_path = (
        case
        / "log.dat"
    )

    log = (
        log_path.read_text(
            errors="replace"
        )
        if log_path.exists()
        else ""
    )

    if proc.returncode != 0:
        raise RuntimeError(
            f"{name} failed rc={proc.returncode}\n"
            f"{proc.stdout}\n"
            f"{log}"
        )

    if (
        "Running on device 0"
        not in log
    ):
        raise RuntimeError(
            f"{name}: real CUDA execution "
            "not confirmed"
        )

    if (
        "END OF THE SIMULATION, everything went OK!"
        not in log
    ):
        raise RuntimeError(
            f"{name}: clean simulation completion "
            "not confirmed"
        )

    pressure_file = (
        case
        / "pressure.dat"
    )

    particle_file = (
        case
        / "particle_stress.dat"
    )

    if not pressure_file.is_file():
        raise RuntimeError(
            f"{name}: pressure.dat missing"
        )

    if not particle_file.is_file():
        raise RuntimeError(
            f"{name}: particle_stress.dat missing"
        )

    return {
        "total": parse_pressure(
            pressure_file
        ),
        "particle": parse_particle_stress(
            particle_file
        ),
    }


# ================================================================
# Analytical expected tensors
# ================================================================

def expected_pd():
    # Protein P1 relative to DNA COM.
    r_center = vsub(
        POSITIONS[1],
        POSITIONS[0],
    )

    # DH interaction itself acts from DNA BACK -> protein.
    r_back = vsub(
        POSITIONS[1],
        BACK_ABS,
    )

    reduced_protein_charge = (
        1.0
        / DNA_EFFECTIVE_CHARGE_E
    )

    charge_product = (
        -reduced_protein_charge
    )

    force_on_protein = (
        dh_force_on_second(
            r_back,
            charge_product,
        )
    )

    total = outer_stress(
        r_center,
        force_on_protein,
    )

    particle = [
        zero_tensor()
        for _ in range(N)
    ]

    half = scale_tensor(
        total,
        0.5,
    )

    # DNA 0 and protein P1 1.
    particle[0] = half
    particle[1] = half

    return (
        total,
        tuple(particle),
    )


def expected_pp():
    r13 = vsub(
        POSITIONS[3],
        POSITIONS[1],
    )

    reduced = (
        1.0
        / DNA_EFFECTIVE_CHARGE_E
    )

    charge_product = (
        reduced
        * reduced
    )

    force_on_p3 = (
        dh_force_on_second(
            r13,
            charge_product,
        )
    )

    total = outer_stress(
        r13,
        force_on_p3,
    )

    particle = [
        zero_tensor()
        for _ in range(N)
    ]

    half = scale_tensor(
        total,
        0.5,
    )

    # P1 global index 1, P3 global index 3.
    particle[1] = half
    particle[3] = half

    return (
        total,
        tuple(particle),
    )


# ================================================================
# Assertions
# ================================================================

def tensor_close(
    actual,
    expected,
    *,
    atol=2.5e-6,
    rtol=4.0e-3,
):
    if len(actual) != 6 or len(expected) != 6:
        raise RuntimeError("CUDA tensor shape mismatch")
    for i, (a, e) in enumerate(
        zip(
            actual,
            expected,
        )
    ):
        if not (math.isfinite(a) and math.isfinite(e)):
            raise RuntimeError(f"CUDA tensor nonfinite: {a}, {e}")
        if (
            abs(a - e)
            >
            atol
            +
            rtol
            * abs(e)
        ):
            raise RuntimeError(
                f"tensor component {i}: "
                f"actual={a:.16e} "
                f"expected={e:.16e}"
            )


def assert_tensor(
    label,
    actual,
    expected,
):
    try:
        tensor_close(
            actual,
            expected,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            f"{label}: {exc}"
        ) from exc


def particle_delta(
    on,
    off,
):
    return tuple(
        sub_tensors(
            on[i],
            off[i],
        )
        for i in range(N)
    )


def sum_particle(
    particles,
):
    total = zero_tensor()

    for t in particles:
        total = add_tensors(
            total,
            t,
        )

    return total


def max_tensor_delta(
    a,
    b,
):
    return max(
        abs(x - y)
        for x, y in zip(a, b)
    )


def max_particle_delta(
    a,
    b,
):
    return max(
        abs(x - y)
        for ta, tb in zip(a, b)
        for x, y in zip(ta, tb)
    )


def run_direct_case(
    root,
    *,
    mode,
    kind,
):
    if kind == "PD":
        charges = (
            +1.0,
            0.0,
            0.0,
        )

        groups = (
            0,
            1,
            2,
        )

        expected_total, expected_particle = (
            expected_pd()
        )

        pd = True
        pp = False

    elif kind == "PP":
        charges = (
            +1.0,
            0.0,
            +1.0,
        )

        groups = (
            0,
            1,
            2,
        )

        expected_total, expected_particle = (
            expected_pp()
        )

        pd = False
        pp = True

    else:
        raise RuntimeError(
            f"Unknown kind {kind}"
        )

    off = run_case(
        root,
        f"{mode}_{kind}_OFF",
        mode=mode,
        pd=False,
        pp=False,
        charges=charges,
        groups=groups,
    )

    on = run_case(
        root,
        f"{mode}_{kind}_ON",
        mode=mode,
        pd=pd,
        pp=pp,
        charges=charges,
        groups=groups,
    )

    total_delta = sub_tensors(
        on["total"],
        off["total"],
    )

    particle_diff = particle_delta(
        on["particle"],
        off["particle"],
    )

    assert_tensor(
        f"{mode}/{kind}/TOTAL",
        total_delta,
        expected_total,
    )

    for i in range(N):
        assert_tensor(
            f"{mode}/{kind}/PARTICLE/{i}",
            particle_diff[i],
            expected_particle[i],
        )

    # Sum of particle virials must recover the total raw virial.
    particle_sum = sum_particle(
        particle_diff
    )

    assert_tensor(
        f"{mode}/{kind}/PARTICLE_SUM",
        particle_sum,
        total_delta,
    )

    # Explicit half-split equality for the interacting pair.
    if kind == "PD":
        a = particle_diff[0]
        b = particle_diff[1]

    else:
        a = particle_diff[1]
        b = particle_diff[3]

    assert_tensor(
        f"{mode}/{kind}/HALF_SPLIT_EQUAL",
        a,
        b,
    )

    assert_tensor(
        f"{mode}/{kind}/HALF_SPLIT_VALUE",
        a,
        scale_tensor(
            expected_total,
            0.5,
        ),
    )

    print(
        "STRESS_CASE_PASS",
        mode,
        kind,
        "TOTAL=",
        total_delta,
        "PARTICLES=",
        particle_diff,
    )

    return {
        "total": total_delta,
        "particle": particle_diff,
    }


def run_exclusion_case(
    root,
    *,
    mode,
    exclusion,
):
    if exclusion == "GROUP":

        groups = (
            0,
            1,
            0,
        )

        connect_13 = False

    elif exclusion == "ANM":

        groups = (
            0,
            1,
            2,
        )

        connect_13 = True

    else:
        raise RuntimeError(
            f"Unknown exclusion {exclusion}"
        )

    charges = (
        +1.0,
        0.0,
        +1.0,
    )

    off = run_case(
        root,
        f"{mode}_PP_{exclusion}_OFF",
        mode=mode,
        pd=False,
        pp=False,
        charges=charges,
        groups=groups,
        exclude_group=True,
        connect_13=connect_13,
    )

    on = run_case(
        root,
        f"{mode}_PP_{exclusion}_ON",
        mode=mode,
        pd=False,
        pp=True,
        charges=charges,
        groups=groups,
        exclude_group=True,
        connect_13=connect_13,
    )

    total_delta = sub_tensors(
        on["total"],
        off["total"],
    )

    particle_diff = particle_delta(
        on["particle"],
        off["particle"],
    )

    assert_tensor(
        f"{mode}/PP_{exclusion}/TOTAL_ZERO",
        total_delta,
        zero_tensor(),
    )

    for i in range(N):
        assert_tensor(
            f"{mode}/PP_{exclusion}/PARTICLE_{i}_ZERO",
            particle_diff[i],
            zero_tensor(),
        )

    print(
        "STRESS_EXCLUSION_PASS",
        mode,
        exclusion,
    )

    return {
        "total": total_delta,
        "particle": particle_diff,
    }


# ================================================================
# Main matrix
# ================================================================

def main():

    pd_total_expected, _ = (
        expected_pd()
    )

    pp_total_expected, _ = (
        expected_pp()
    )

    print(
        "EXPECTED_PD_STRESS",
        pd_total_expected,
    )

    print(
        "EXPECTED_PP_STRESS",
        pp_total_expected,
    )

    # Sanity against previously qualified CPU values.
    if not (
        pd_total_expected[2] < 0.0
        and
        pd_total_expected[4] > 0.0
        and
        pd_total_expected[5] < 0.0
    ):
        raise RuntimeError(
            "PD analytical stress sign sanity failed"
        )

    if not (
        pp_total_expected[0] > 0.0
        and
        all(
            abs(pp_total_expected[i])
            < 1e-14
            for i in range(
                1,
                6,
            )
        )
    ):
        raise RuntimeError(
            "PP analytical stress sign sanity failed"
        )

    print(
        "ANALYTICAL_STRESS_SIGN_SANITY=PASS"
    )

    results = {}

    with tempfile.TemporaryDirectory(
        prefix="oxdna_cuda_stress25_"
    ) as td:

        root = Path(td)

        for mode in (
            "DNANM",
            "DNACT",
        ):

            results[
                (
                    mode,
                    "PD",
                )
            ] = run_direct_case(
                root,
                mode=mode,
                kind="PD",
            )

            results[
                (
                    mode,
                    "PP",
                )
            ] = run_direct_case(
                root,
                mode=mode,
                kind="PP",
            )

            results[
                (
                    mode,
                    "GROUP",
                )
            ] = run_exclusion_case(
                root,
                mode=mode,
                exclusion="GROUP",
            )

            results[
                (
                    mode,
                    "ANM",
                )
            ] = run_exclusion_case(
                root,
                mode=mode,
                exclusion="ANM",
            )

    # ============================================================
    # DNANM / DNACT parity
    # ============================================================

    for label in (
        "PD",
        "PP",
        "GROUP",
        "ANM",
    ):

        a = results[
            (
                "DNANM",
                label,
            )
        ]

        b = results[
            (
                "DNACT",
                label,
            )
        ]

        td = max_tensor_delta(
            a["total"],
            b["total"],
        )

        pd = max_particle_delta(
            a["particle"],
            b["particle"],
        )

        print(
            "STRESS_DNANM_DNACT_PARITY",
            label,
            f"total_delta={td:.16e}",
            f"particle_delta={pd:.16e}",
        )

        if td > 2.5e-6:
            raise RuntimeError(
                f"{label}: DNANM/DNACT total "
                f"stress mismatch {td}"
            )

        if pd > 2.5e-6:
            raise RuntimeError(
                f"{label}: DNANM/DNACT particle "
                f"stress mismatch {pd}"
            )

    print("CUDA_PD_TOTAL_STRESS=PASS")
    print("CUDA_PP_TOTAL_STRESS=PASS")

    print("CUDA_PD_PARTICLE_STRESS=PASS")
    print("CUDA_PP_PARTICLE_STRESS=PASS")

    print("CUDA_PD_HALF_SPLIT_STRESS=PASS")
    print("CUDA_PP_HALF_SPLIT_STRESS=PASS")

    print("CUDA_PARTICLE_SUM_EQUALS_TOTAL=PASS")

    print("CUDA_PD_OFFDIAGONAL_STRESS=PASS")
    print("CUDA_PP_DIAGONAL_STRESS=PASS")

    print("CUDA_GROUP_EXCLUSION_STRESS_ZERO=PASS")
    print("CUDA_ANM_EXCLUSION_STRESS_ZERO=PASS")

    print("CUDA_DNANM_DNACT_STRESS_PARITY=PASS")

    print("CUDA_ANALYTICAL_STRESS_GATE=PASS")


if __name__ == "__main__":
    main()
