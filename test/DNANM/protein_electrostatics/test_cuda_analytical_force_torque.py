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
# Numerical / physical constants
# ================================================================

N = 4

DT = 1.0e-4

DNA_EFFECTIVE_CHARGE_E = 0.815

DH_PREF = 0.0543
DH_LAMBDA = 0.3616455

DH_RHIGH = 3.0 * DH_LAMBDA

# Same smoothing construction as oxDNA2.
DH_B = -(
    math.exp(
        -DH_RHIGH
        / DH_LAMBDA
    )
    * DH_PREF
    * DH_PREF
    * (
        DH_RHIGH
        + DH_LAMBDA
    ) ** 2
) / (
    -4.0
    * DH_RHIGH**3
    * DH_LAMBDA**2
    * DH_PREF
)

DH_RC = (
    DH_RHIGH
    * (
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
        * (
            DH_RHIGH
            + DH_LAMBDA
        )
    )
)

# Identity nucleotide orientation.
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

# Keep angular momentum non-zero so CUDA's quaternion update
# never sees |L| = 0. It is tiny enough that the one-step
# orientation change is negligible at mixed precision.
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
# Vector utilities
# ================================================================

def vadd(a, b):
    return tuple(
        x + y
        for x, y in zip(a, b)
    )


def vsub(a, b):
    return tuple(
        x - y
        for x, y in zip(a, b)
    )


def vscale(a, k):
    return tuple(
        k * x
        for x in a
    )


def vnorm(a):
    return math.sqrt(
        sum(
            x * x
            for x in a
        )
    )


def cross(a, b):
    return (
        a[1] * b[2]
        - a[2] * b[1],

        a[2] * b[0]
        - a[0] * b[2],

        a[0] * b[1]
        - a[1] * b[0],
    )


def distance(a, b):
    return vnorm(
        vsub(a, b)
    )


def all_zero_vectors():
    return [
        (0.0, 0.0, 0.0)
        for _ in range(N)
    ]


# ================================================================
# Exact analytical DH used by the new CUDA term
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
            -r / DH_LAMBDA
        )

        scalar = (
            DH_PREF
            * exponential
            * (
                1.0
                / (
                    DH_LAMBDA
                    * r
                )
                +
                1.0
                / (
                    r * r
                )
            )
            * charge_product
        )

    else:

        scalar = (
            -2.0
            * DH_B
            * (
                r
                - DH_RC
            )
            * charge_product
        )

    return vscale(
        rhat,
        scalar,
    )


# ================================================================
# Micro-system construction
# ================================================================

def make_positions(
    kind,
    separation,
):
    # P1 is defined from the nucleotide BACK site for PD.
    p1 = (
        BACK_ABS[0],
        BACK_ABS[1],
        BACK_ABS[2]
        + (
            separation
            if kind == "PD"
            else 1.02
        ),
    )

    p2 = (
        12.0,
        12.0,
        12.0,
    )

    if kind == "PP":
        p3 = (
            p1[0]
            + separation,
            p1[1],
            p1[2],
        )
    else:
        p3 = (
            p1[0]
            + 0.8,
            p1[1],
            p1[2],
        )

    return (
        DNA_POS,
        p1,
        p2,
        p3,
    )


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


def conf_text(positions):
    return (
        "t = 0\n"
        "b = 20 20 20\n"
        "E = 0 0 0\n"
        +
        "\n".join(
            particle_line(p)
            for p in positions
        )
        +
        "\n"
    )


def charges_text(
    charges,
    groups,
):
    n_groups = len(
        set(groups)
    )

    return f"""# DNANM_ES_CHARGES 1
# index_space = protein_local
# index_base = 0
# charge_model = cuda-analytical-force-torque
# charge_source = analytical
# n_protein_beads = 3
# n_groups = {n_groups}
# columns = protein_index topology_strand residue charge_e group_id
0 -1 K {charges[0]:.16g} {groups[0]}
1 -2 K {charges[1]:.16g} {groups[1]}
2 -3 K {charges[2]:.16g} {groups[2]}
"""


def springs_text(
    mode,
    positions,
    connect_test_pair,
):
    p1 = positions[1]
    p2 = positions[2]
    p3 = positions[3]

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

    # Direct P1-P3 ANM connection for the dedicated exclusion
    # test. k=0 means physically inert but topologically affected.
    if connect_test_pair:
        lines.append(
            f"1 3 {d13:.16g} s 0.0"
        )

    return (
        "\n".join(lines)
        + "\n"
    )


def input_text(
    mode,
    *,
    pd,
    pp,
    half,
    charge_scale,
    pd_scale,
    pp_scale,
    dna_ref,
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
dh_half_charged_ends = {"true" if half else "false"}

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

print_conf_interval = 1
print_energy_every = 1

protein_dna_electrostatics = {"true" if pd else "false"}
protein_protein_electrostatics = {"true" if pp else "false"}

protein_charge_file = protein_charges.dat

protein_charge_scale = {charge_scale:.16g}
protein_dna_dh_scale = {pd_scale:.16g}
protein_protein_dh_scale = {pp_scale:.16g}

dna_effective_charge_e = {dna_ref:.16g}

exclude_intragroup_electrostatics = {"true" if exclude_group else "false"}
"""


# ================================================================
# Runtime / state extraction
# ================================================================

def parse_state(path):
    lines = [
        x.strip()
        for x in path.read_text().splitlines()
        if x.strip()
    ]

    if len(lines) < 3 + N:
        raise RuntimeError(
            f"Malformed configuration: {path}"
        )

    rows = []

    for line in lines[
        3:
        3 + N
    ]:
        vals = tuple(
            float(x)
            for x in line.split()
        )

        if len(vals) != 15:
            raise RuntimeError(
                f"Expected 15 fields, got "
                f"{len(vals)} in {line}"
            )

        rows.append(vals)

    return tuple(rows)


def run_case(
    root,
    name,
    *,
    mode,
    kind,
    separation,
    pd,
    pp,
    charges,
    groups,
    half=False,
    charge_scale=1.0,
    pd_scale=1.0,
    pp_scale=1.0,
    dna_ref=DNA_EFFECTIVE_CHARGE_E,
    exclude_group=True,
    connect_test_pair=False,
):
    case = root / name
    case.mkdir()

    positions = make_positions(
        kind,
        separation,
    )

    (case / "system.top").write_text(
        TOP
    )

    (case / "init.dat").write_text(
        conf_text(
            positions
        )
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
            positions,
            connect_test_pair,
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
            half=half,
            charge_scale=charge_scale,
            pd_scale=pd_scale,
            pp_scale=pp_scale,
            dna_ref=dna_ref,
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
            f"{name}: CUDA device execution "
            "not confirmed"
        )

    if (
        "END OF THE SIMULATION, everything went OK!"
        not in log
    ):
        raise RuntimeError(
            f"{name}: simulation completion "
            "not confirmed"
        )

    state = parse_state(
        case
        / "last_conf.dat"
    )

    return {
        "state": state,
        "positions": positions,
    }


def response(
    off,
    on,
):
    # Effective device force from one velocity-Verlet step:
    #
    # delta(v) / dt.
    #
    # Effective device torque:
    #
    # delta(L) / dt.
    #
    # With DT=1e-4 the translational/orientational displacement
    # is below mixed-precision coordinate resolution for this
    # micro-system, so the force is evaluated at effectively the
    # same geometry in ON and OFF.
    forces = []
    torques = []

    max_pos_delta = 0.0
    max_ori_delta = 0.0

    for a, b in zip(
        off["state"],
        on["state"],
    ):
        max_pos_delta = max(
            max_pos_delta,
            *[
                abs(
                    b[i]
                    - a[i]
                )
                for i in range(3)
            ],
        )

        max_ori_delta = max(
            max_ori_delta,
            *[
                abs(
                    b[i]
                    - a[i]
                )
                for i in range(
                    3,
                    9,
                )
            ],
        )

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

    return {
        "forces": tuple(forces),
        "torques": tuple(torques),
        "max_pos_delta": max_pos_delta,
        "max_ori_delta": max_ori_delta,
    }


# ================================================================
# Expected forces / torques
# ================================================================

def expected_pd(
    *,
    separation,
    q1,
    half,
    charge_scale,
    pd_scale,
    dna_ref,
):
    reduced_q = (
        charge_scale
        * q1
        / dna_ref
    )

    charge_product = (
        -pd_scale
        * reduced_q
    )

    if half:
        charge_product *= 0.5

    rback = (
        0.0,
        0.0,
        separation,
    )

    force_on_protein = (
        dh_force_on_second(
            rback,
            charge_product,
        )
    )

    force_on_dna = vscale(
        force_on_protein,
        -1.0,
    )

    forces = all_zero_vectors()

    forces[0] = force_on_dna
    forces[1] = force_on_protein

    torques = all_zero_vectors()

    torques[0] = cross(
        BACK,
        force_on_dna,
    )

    return (
        tuple(forces),
        tuple(torques),
    )


def expected_pp(
    *,
    separation,
    q1,
    q3,
    charge_scale,
    pp_scale,
    dna_ref,
    excluded,
):
    if excluded:
        return (
            tuple(
                all_zero_vectors()
            ),
            tuple(
                all_zero_vectors()
            ),
        )

    q1_reduced = (
        charge_scale
        * q1
        / dna_ref
    )

    q3_reduced = (
        charge_scale
        * q3
        / dna_ref
    )

    charge_product = (
        pp_scale
        *
        q1_reduced
        *
        q3_reduced
    )

    r13 = (
        separation,
        0.0,
        0.0,
    )

    force_on_p3 = (
        dh_force_on_second(
            r13,
            charge_product,
        )
    )

    force_on_p1 = vscale(
        force_on_p3,
        -1.0,
    )

    forces = all_zero_vectors()

    forces[1] = force_on_p1
    forces[3] = force_on_p3

    torques = all_zero_vectors()

    return (
        tuple(forces),
        tuple(torques),
    )


# ================================================================
# Assertions
# ================================================================

def scalar_close(
    actual,
    expected,
    *,
    atol=5.0e-6,
    rtol=4.0e-3,
):
    return (
        abs(
            actual
            - expected
        )
        <=
        atol
        +
        rtol
        * abs(expected)
    )


def assert_vectors(
    label,
    actual,
    expected,
    *,
    atol=5.0e-6,
    rtol=4.0e-3,
):
    for i, (
        avec,
        evec,
    ) in enumerate(
        zip(
            actual,
            expected,
        )
    ):
        for axis, (
            a,
            e,
        ) in enumerate(
            zip(
                avec,
                evec,
            )
        ):
            if not scalar_close(
                a,
                e,
                atol=atol,
                rtol=rtol,
            ):
                raise RuntimeError(
                    f"{label}: particle={i} "
                    f"axis={axis} "
                    f"actual={a:.16e} "
                    f"expected={e:.16e}"
                )


def max_vector_difference(
    a,
    b,
):
    return max(
        abs(x - y)
        for va, vb in zip(a, b)
        for x, y in zip(va, vb)
    )


def assert_newton(
    label,
    forces,
):
    total = (
        sum(v[0] for v in forces),
        sum(v[1] for v in forces),
        sum(v[2] for v in forces),
    )

    err = vnorm(total)

    print(
        "NEWTON_III",
        label,
        f"|sumF|={err:.16e}",
    )

    if err > 2.0e-5:
        raise RuntimeError(
            f"{label}: Newton III failed, "
            f"sumF={total}"
        )


def run_pair(
    root,
    mode,
    spec,
):
    common = dict(
        mode=mode,
        kind=spec["kind"],
        separation=spec["separation"],
        charges=spec["charges"],
        groups=spec["groups"],
        half=spec.get(
            "half",
            False,
        ),
        charge_scale=spec.get(
            "charge_scale",
            1.0,
        ),
        pd_scale=spec.get(
            "pd_scale",
            1.0,
        ),
        pp_scale=spec.get(
            "pp_scale",
            1.0,
        ),
        dna_ref=spec.get(
            "dna_ref",
            DNA_EFFECTIVE_CHARGE_E,
        ),
        exclude_group=spec.get(
            "exclude_group",
            True,
        ),
        connect_test_pair=spec.get(
            "connect_test_pair",
            False,
        ),
    )

    off = run_case(
        root,
        f"{mode}_{spec['name']}_OFF",
        pd=False,
        pp=False,
        **common,
    )

    on = run_case(
        root,
        f"{mode}_{spec['name']}_ON",
        pd=(
            spec["kind"]
            == "PD"
        ),
        pp=(
            spec["kind"]
            == "PP"
        ),
        **common,
    )

    resp = response(
        off,
        on,
    )

    # ON/OFF positions and orientation should remain effectively
    # frozen over this single tiny step.
    if (
        resp["max_pos_delta"]
        > 2.0e-7
    ):
        raise RuntimeError(
            f"{mode}/{spec['name']}: "
            f"geometry moved too far: "
            f"{resp['max_pos_delta']}"
        )

    if (
        resp["max_ori_delta"]
        > 2.0e-7
    ):
        raise RuntimeError(
            f"{mode}/{spec['name']}: "
            f"orientation moved too far: "
            f"{resp['max_ori_delta']}"
        )

    if spec["kind"] == "PD":

        expected_f, expected_t = (
            expected_pd(
                separation=spec[
                    "separation"
                ],
                q1=spec["charges"][0],
                half=spec.get(
                    "half",
                    False,
                ),
                charge_scale=spec.get(
                    "charge_scale",
                    1.0,
                ),
                pd_scale=spec.get(
                    "pd_scale",
                    1.0,
                ),
                dna_ref=spec.get(
                    "dna_ref",
                    DNA_EFFECTIVE_CHARGE_E,
                ),
            )
        )

    else:

        expected_f, expected_t = (
            expected_pp(
                separation=spec[
                    "separation"
                ],
                q1=spec["charges"][0],
                q3=spec["charges"][2],
                charge_scale=spec.get(
                    "charge_scale",
                    1.0,
                ),
                pp_scale=spec.get(
                    "pp_scale",
                    1.0,
                ),
                dna_ref=spec.get(
                    "dna_ref",
                    DNA_EFFECTIVE_CHARGE_E,
                ),
                excluded=spec.get(
                    "expected_excluded",
                    False,
                ),
            )
        )

    assert_vectors(
        f"{mode}/{spec['name']}/FORCE",
        resp["forces"],
        expected_f,
        atol=5.0e-6,
        rtol=4.0e-3,
    )

    assert_vectors(
        f"{mode}/{spec['name']}/TORQUE",
        resp["torques"],
        expected_t,
        atol=5.0e-6,
        rtol=6.0e-3,
    )

    assert_newton(
        f"{mode}/{spec['name']}",
        resp["forces"],
    )

    print(
        "CASE_PASS",
        mode,
        spec["name"],
        "F=",
        resp["forces"],
        "T=",
        resp["torques"],
    )

    return resp


# ================================================================
# Qualification matrix
# ================================================================

SPECS = [
    # ---------------- PD direct branch / signs ----------------
    {
        "name": "PD_DIRECT_ATTRACTION",
        "kind": "PD",
        "separation": 1.02,
        "charges": (+1.0, 0.0, 0.0),
        "groups": (0, 1, 2),
    },
    {
        "name": "PD_DIRECT_REPULSION",
        "kind": "PD",
        "separation": 1.02,
        "charges": (-1.0, 0.0, 0.0),
        "groups": (0, 1, 2),
    },

    # ---------------- PD smoothing / cutoff ----------------
    {
        "name": "PD_SMOOTHING",
        "kind": "PD",
        "separation": 1.30,
        "charges": (+1.0, 0.0, 0.0),
        "groups": (0, 1, 2),
    },
    {
        "name": "PD_CUTOFF",
        "kind": "PD",
        "separation": 1.80,
        "charges": (+1.0, 0.0, 0.0),
        "groups": (0, 1, 2),
    },

    # ---------------- PD controls ----------------
    {
        "name": "PD_HALF_CHARGED_END",
        "kind": "PD",
        "separation": 1.02,
        "charges": (+1.0, 0.0, 0.0),
        "groups": (0, 1, 2),
        "half": True,
    },
    {
        "name": "PD_DH_SCALE_1P7",
        "kind": "PD",
        "separation": 1.02,
        "charges": (+1.0, 0.0, 0.0),
        "groups": (0, 1, 2),
        "pd_scale": 1.7,
    },
    {
        "name": "PD_CHARGE_SCALE_HALF",
        "kind": "PD",
        "separation": 1.02,
        "charges": (+1.0, 0.0, 0.0),
        "groups": (0, 1, 2),
        "charge_scale": 0.5,
    },
    {
        "name": "PD_DNA_REFERENCE_DOUBLE",
        "kind": "PD",
        "separation": 1.02,
        "charges": (+1.0, 0.0, 0.0),
        "groups": (0, 1, 2),
        "dna_ref": 2.0 * DNA_EFFECTIVE_CHARGE_E,
    },

    # ---------------- PP direct / signs ----------------
    {
        "name": "PP_DIRECT_REPULSION",
        "kind": "PP",
        "separation": 0.80,
        "charges": (+1.0, 0.0, +1.0),
        "groups": (0, 1, 2),
    },
    {
        "name": "PP_DIRECT_ATTRACTION",
        "kind": "PP",
        "separation": 0.80,
        "charges": (+1.0, 0.0, -1.0),
        "groups": (0, 1, 2),
    },

    # ---------------- PP smoothing / cutoff ----------------
    {
        "name": "PP_SMOOTHING",
        "kind": "PP",
        "separation": 1.30,
        "charges": (+1.0, 0.0, +1.0),
        "groups": (0, 1, 2),
    },
    {
        "name": "PP_CUTOFF",
        "kind": "PP",
        "separation": 1.80,
        "charges": (+1.0, 0.0, +1.0),
        "groups": (0, 1, 2),
    },

    # ---------------- PP scaling ----------------
    {
        "name": "PP_DH_SCALE_0P4",
        "kind": "PP",
        "separation": 0.80,
        "charges": (+1.0, 0.0, +1.0),
        "groups": (0, 1, 2),
        "pp_scale": 0.4,
    },
    {
        "name": "PP_CHARGE_SCALE_HALF",
        "kind": "PP",
        "separation": 0.80,
        "charges": (+1.0, 0.0, +1.0),
        "groups": (0, 1, 2),
        "charge_scale": 0.5,
    },
    {
        "name": "PP_DNA_REFERENCE_DOUBLE",
        "kind": "PP",
        "separation": 0.80,
        "charges": (+1.0, 0.0, +1.0),
        "groups": (0, 1, 2),
        "dna_ref": 2.0 * DNA_EFFECTIVE_CHARGE_E,
    },

    # ---------------- group semantics ----------------
    {
        "name": "PP_GROUP_EXCLUDED",
        "kind": "PP",
        "separation": 0.80,
        "charges": (+1.0, 0.0, +1.0),
        "groups": (0, 1, 0),
        "exclude_group": True,
        "expected_excluded": True,
    },
    {
        "name": "PP_GROUP_ALLOWED",
        "kind": "PP",
        "separation": 0.80,
        "charges": (+1.0, 0.0, +1.0),
        "groups": (0, 1, 0),
        "exclude_group": False,
        "expected_excluded": False,
    },

    # ---------------- ANM exclusion ----------------
    {
        "name": "PP_ANM_EXCLUDED",
        "kind": "PP",
        "separation": 0.80,
        "charges": (+1.0, 0.0, +1.0),
        "groups": (0, 1, 2),
        "connect_test_pair": True,
        "expected_excluded": True,
    },
]


def main():
    print(
        "DH_PARAMETERS",
        f"lambda={DH_LAMBDA:.16e}",
        f"RHIGH={DH_RHIGH:.16e}",
        f"B={DH_B:.16e}",
        f"RC={DH_RC:.16e}",
    )

    if not (
        1.02
        < DH_RHIGH
        < 1.30
        < DH_RC
        < 1.80
    ):
        raise RuntimeError(
            "Analytical test distances do not span "
            "the intended DH branches"
        )

    print(
        "DH_BRANCH_GEOMETRY=PASS"
    )

    results = {}

    with tempfile.TemporaryDirectory(
        prefix="oxdna_cuda_analytic24_"
    ) as td:

        root = Path(td)

        for mode in (
            "DNANM",
            "DNACT",
        ):
            for spec in SPECS:
                result = run_pair(
                    root,
                    mode,
                    spec,
                )

                results[
                    (
                        mode,
                        spec["name"],
                    )
                ] = result

    # ============================================================
    # DNANM <-> DNACT shared CUDA physics parity
    # ============================================================

    for spec in SPECS:
        name = spec["name"]

        a = results[
            (
                "DNANM",
                name,
            )
        ]

        b = results[
            (
                "DNACT",
                name,
            )
        ]

        fdelta = max_vector_difference(
            a["forces"],
            b["forces"],
        )

        tdelta = max_vector_difference(
            a["torques"],
            b["torques"],
        )

        print(
            "DNANM_DNACT_PARITY",
            name,
            f"force_delta={fdelta:.16e}",
            f"torque_delta={tdelta:.16e}",
        )

        if fdelta > 5.0e-6:
            raise RuntimeError(
                f"{name}: DNANM/DNACT force "
                f"mismatch {fdelta}"
            )

        if tdelta > 5.0e-6:
            raise RuntimeError(
                f"{name}: DNANM/DNACT torque "
                f"mismatch {tdelta}"
            )

    # ============================================================
    # Final qualification markers
    # ============================================================

    print("CUDA_ANALYTIC_PD_FORCE=PASS")
    print("CUDA_ANALYTIC_PP_FORCE=PASS")

    print("CUDA_PD_ATTRACTION_REPULSION=PASS")
    print("CUDA_PP_ATTRACTION_REPULSION=PASS")

    print("CUDA_PD_BACK_TORQUE=PASS")
    print("CUDA_NEWTON_III=PASS")

    print("CUDA_DH_DIRECT_BRANCH=PASS")
    print("CUDA_DH_SMOOTHING_BRANCH=PASS")
    print("CUDA_DH_CUTOFF=PASS")

    print("CUDA_HALF_CHARGED_ENDS=PASS")

    print("CUDA_PD_PP_DH_SCALING=PASS")
    print("CUDA_PROTEIN_CHARGE_SCALE=PASS")
    print("CUDA_DNA_EFFECTIVE_CHARGE_NORMALISATION=PASS")

    print("CUDA_PP_GROUP_EXCLUSION=PASS")
    print("CUDA_PP_GROUP_OVERRIDE=PASS")
    print("CUDA_PP_ANM_EXCLUSION=PASS")

    print("CUDA_DNANM_DNACT_FORCE_TORQUE_PARITY=PASS")

    print("CUDA_ANALYTICAL_FORCE_TORQUE_GATE=PASS")


if __name__ == "__main__":
    main()
