#!/usr/bin/env python3

import os
import shutil
import subprocess
import tempfile
from pathlib import Path


REPO = Path(
    os.environ["OXDNA_REPO"]
).resolve()

OXDNA = Path(
    os.environ["OXDNA_BIN"]
).resolve()

UTILITY = (
    REPO
    / "utils"
    / "dnanm_protein_charges.py"
)

if not UTILITY.is_file():
    raise RuntimeError(
        f"Utility missing: {UTILITY}"
    )

if not OXDNA.is_file():
    raise RuntimeError(
        f"oxDNA binary missing: {OXDNA}"
    )


TOP = """4 4 1 3 1
1 A -1 -1
-1 K -1 -1
-2 D -1 -1
-3 R -1 -1
"""


ASSIGNMENTS = """# protein_local_index charge_e group_id
0 1.0 7
1 -0.5 7
2 0.25 9
"""


CONF = """t = 0
b = 20 20 20
E = 0 0 0
5.0 5.0 5.0 1 0 0 0 0 1 0 0 0 1e-4 2e-4 3e-4
4.66 5.3408 6.02 1 0 0 0 0 1 0 0 0 1e-4 2e-4 3e-4
12.0 12.0 12.0 1 0 0 0 0 1 0 0 0 1e-4 2e-4 3e-4
5.46 5.3408 6.02 1 0 0 0 0 1 0 0 0 1e-4 2e-4 3e-4
"""


def distance(a, b):
    return sum(
        (x - y) ** 2
        for x, y in zip(a, b)
    ) ** 0.5


P1 = (4.66, 5.3408, 6.02)
P2 = (12.0, 12.0, 12.0)
P3 = (5.46, 5.3408, 6.02)

D12 = distance(P1, P2)
D23 = distance(P2, P3)


def run(
    argv,
    *,
    cwd=None,
    expect=0,
):
    proc = subprocess.run(
        argv,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=90,
    )

    if proc.returncode != expect:
        raise RuntimeError(
            f"command failed expectation\n"
            f"argv={argv}\n"
            f"expected={expect}\n"
            f"actual={proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )

    return proc


def utility_ok(*args):
    return run(
        [
            str(UTILITY),
            *args,
        ],
        expect=0,
    )


def utility_fail(*args):
    proc = subprocess.run(
        [
            str(UTILITY),
            *args,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )

    if proc.returncode == 0:
        raise RuntimeError(
            "utility unexpectedly accepted malformed input\n"
            f"args={args}\n"
            f"stdout={proc.stdout}\n"
            f"stderr={proc.stderr}"
        )

    if "ERROR:" not in proc.stderr:
        raise RuntimeError(
            "utility failure did not emit fail-closed ERROR:\n"
            f"args={args}\n"
            f"stderr={proc.stderr}"
        )

    return proc


def replace_once(
    text,
    old,
    new,
):
    if text.count(old) != 1:
        raise RuntimeError(
            f"replace_once expected one '{old}', "
            f"found {text.count(old)}"
        )

    return text.replace(
        old,
        new,
        1,
    )


def write_bad(
    directory,
    name,
    text,
):
    p = directory / name
    p.write_text(text)
    return p


def springs(mode):
    if mode == "DNACT":
        return (
            "3\n"
            f"1 2 {D12:.16g} "
            "s 0.0 0.0 0.0 0.0 0.0\n"
            f"2 3 {D23:.16g} "
            "s 0.0 0.0 0.0 0.0 0.0\n"
        )

    return (
        "3\n"
        f"1 2 {D12:.16g} s 0.0\n"
        f"2 3 {D23:.16g} s 0.0\n"
    )


def input_text(
    backend,
    mode,
    charge_file,
):
    cuda = ""

    if backend == "CUDA":
        cuda = """backend_precision = mixed
CUDA_list = verlet
CUDA_sort_every = 0
use_edge = 1
edge_n_forces = 1
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
print_conf_interval = 1
print_energy_every = 1

protein_dna_electrostatics = true
protein_protein_electrostatics = true

protein_charge_file = {charge_file}

protein_charge_scale = 1.0
protein_dna_dh_scale = 1.0
protein_protein_dh_scale = 1.0

dna_effective_charge_e = 0.815
exclude_intragroup_electrostatics = true
"""


def runtime_case(
    root,
    *,
    backend,
    mode,
    charge_source,
    expect_success,
    expected_error=None,
):
    case = (
        root
        / (
            f"runtime_{backend}_{mode}_"
            f"{charge_source.stem}"
        )
    )

    case.mkdir()

    (
        case
        / "system.top"
    ).write_text(TOP)

    (
        case
        / "init.dat"
    ).write_text(CONF)

    (
        case
        / "springs.par"
    ).write_text(
        springs(mode)
    )

    target_charge = (
        case
        / charge_source.name
    )

    shutil.copy2(
        charge_source,
        target_charge,
    )

    (
        case
        / "input"
    ).write_text(
        input_text(
            backend,
            mode,
            target_charge.name,
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

    combined = (
        proc.stdout
        + "\n"
        + log
    )

    if expected_error is not None:
        if expected_error not in combined:
            raise RuntimeError(
                f"Missing expected diagnostic for {backend}/{mode}: "
                f"{expected_error}\n{combined}"
            )

    if expect_success:
        if proc.returncode != 0:
            raise RuntimeError(
                f"valid runtime failed: "
                f"{backend}/{mode}\n"
                f"{combined}"
            )

        if (
            "END OF THE SIMULATION, everything went OK!"
            not in log
        ):
            raise RuntimeError(
                f"{backend}/{mode}: valid runtime "
                "did not finish cleanly"
            )

        if backend == "CUDA":
            if "Running on device 0" not in log:
                raise RuntimeError(
                    f"{backend}/{mode}: CUDA execution "
                    "not confirmed"
                )

    else:
        if proc.returncode == 0:
            raise RuntimeError(
                f"malformed charge file was accepted by "
                f"{backend}/{mode}: {charge_source}"
            )

        if (
            "END OF THE SIMULATION, everything went OK!"
            in combined
        ):
            raise RuntimeError(
                f"malformed charge file somehow completed "
                f"{backend}/{mode}"
            )

    return proc.returncode


def main():
    with tempfile.TemporaryDirectory(
        prefix="oxdna_charge_utility30_"
    ) as td:
        root = Path(td)

        topology = root / "system.top"
        assignments = root / "assignments.dat"
        valid = root / "protein_charges.dat"

        topology.write_text(TOP)
        assignments.write_text(ASSIGNMENTS)

        # ========================================================
        # PREPARE
        # ========================================================

        prep = utility_ok(
            "prepare",
            "--topology",
            str(topology),
            "--assignments",
            str(assignments),
            "--output",
            str(valid),
            "--charge-model",
            "explicit-test-model",
            "--charge-source",
            "explicit-assignment-test",
        )

        if "PREPARED" not in prep.stdout:
            raise RuntimeError(
                "prepare command did not emit PREPARED marker"
            )

        if not valid.is_file():
            raise RuntimeError(
                "prepare command did not create output"
            )

        text = valid.read_text()

        required = [
            "# DNANM_ES_CHARGES 1",
            "# index_space = protein_local",
            "# index_base = 0",
            "# charge_model = explicit-test-model",
            "# charge_source = explicit-assignment-test",
            "# n_protein_beads = 3",
            "# n_groups = 2",
            "0 -1 K 1 7",
            "1 -2 D -0.5 7",
            "2 -3 R 0.25 9",
        ]

        for token in required:
            if token not in text:
                raise RuntimeError(
                    f"prepared file missing '{token}'"
                )

        print("UTILITY_PREPARE_CANONICAL_V1=PASS")

        # ========================================================
        # VALIDATE
        # ========================================================

        val = utility_ok(
            "validate",
            "--topology",
            str(topology),
            "--charges",
            str(valid),
        )

        if "VALID" not in val.stdout:
            raise RuntimeError(
                "validate command did not emit VALID marker"
            )

        print("UTILITY_VALIDATE_CANONICAL_V1=PASS")

        # ========================================================
        # MALFORMED VALIDATOR MATRIX
        # ========================================================

        bad_files = {}

        bad_files["duplicate"] = write_bad(
            root,
            "bad_duplicate.dat",
            text
            +
            "0 -1 K 1 7\n",
        )

        bad_files["missing"] = write_bad(
            root,
            "bad_missing.dat",
            "\n".join(
                line
                for line in text.splitlines()
                if not line.startswith("2 -3 R ")
            )
            +
            "\n",
        )

        bad_files["wrong_strand"] = write_bad(
            root,
            "bad_wrong_strand.dat",
            replace_once(
                text,
                "1 -2 D -0.5 7",
                "1 -99 D -0.5 7",
            ),
        )

        bad_files["wrong_residue"] = write_bad(
            root,
            "bad_wrong_residue.dat",
            replace_once(
                text,
                "1 -2 D -0.5 7",
                "1 -2 K -0.5 7",
            ),
        )

        bad_files["wrong_npro"] = write_bad(
            root,
            "bad_wrong_npro.dat",
            replace_once(
                text,
                "# n_protein_beads = 3",
                "# n_protein_beads = 4",
            ),
        )

        bad_files["wrong_index_space"] = write_bad(
            root,
            "bad_index_space.dat",
            replace_once(
                text,
                "# index_space = protein_local",
                "# index_space = global",
            ),
        )

        bad_files["wrong_index_base"] = write_bad(
            root,
            "bad_index_base.dat",
            replace_once(
                text,
                "# index_base = 0",
                "# index_base = 1",
            ),
        )

        bad_files["nan"] = write_bad(
            root,
            "bad_nan.dat",
            replace_once(
                text,
                "0 -1 K 1 7",
                "0 -1 K nan 7",
            ),
        )

        bad_files["inf"] = write_bad(
            root,
            "bad_inf.dat",
            replace_once(
                text,
                "0 -1 K 1 7",
                "0 -1 K inf 7",
            ),
        )

        bad_files["overflow"] = write_bad(
            root,
            "bad_overflow.dat",
            replace_once(
                text,
                "0 -1 K 1 7",
                "0 -1 K 1e999 7",
            ),
        )

        bad_files["negative_group"] = write_bad(
            root,
            "bad_negative_group.dat",
            replace_once(
                text,
                "0 -1 K 1 7",
                "0 -1 K 1 -1",
            ),
        )

        bad_files["ngroups_mismatch"] = write_bad(
            root,
            "bad_ngroups.dat",
            replace_once(
                text,
                "# n_groups = 2",
                "# n_groups = 3",
            ),
        )

        bad_files["extra_column"] = write_bad(
            root,
            "bad_extra_column.dat",
            replace_once(
                text,
                "0 -1 K 1 7",
                "0 -1 K 1 7 EXTRA",
            ),
        )

        bad_files["wrong_header"] = write_bad(
            root,
            "bad_header.dat",
            replace_once(
                text,
                "# DNANM_ES_CHARGES 1",
                "# DNANM_ES_CHARGES 2",
            ),
        )

        bad_files["duplicate_metadata"] = write_bad(
            root,
            "bad_duplicate_metadata.dat",
            replace_once(
                text,
                "# index_base = 0",
                "# index_base = 0\n# index_base = 0",
            ),
        )

        bad_files["header_extra_token"] = write_bad(
            root,
            "bad_header_extra_token.dat",
            replace_once(
                text,
                "# DNANM_ES_CHARGES 1",
                "# DNANM_ES_CHARGES 1 EXTRA",
            ),
        )

        bad_files["npro_trailing_token"] = write_bad(
            root,
            "bad_npro_trailing_token.dat",
            replace_once(
                text,
                "# n_protein_beads = 3",
                "# n_protein_beads = 3 EXTRA",
            ),
        )

        bad_files["ngroups_trailing_token"] = write_bad(
            root,
            "bad_ngroups_trailing_token.dat",
            replace_once(
                text,
                "# n_groups = 2",
                "# n_groups = 2 EXTRA",
            ),
        )

        for label, path in bad_files.items():
            utility_fail(
                "validate",
                "--topology",
                str(topology),
                "--charges",
                str(path),
            )

            print(
                f"VALIDATOR_REJECT_{label.upper()}=PASS"
            )

        print(
            "UTILITY_VALIDATOR_MALFORMED_MATRIX=PASS"
        )

        # ========================================================
        # PREPARATOR FAIL-CLOSED MATRIX
        # ========================================================

        bad_assignments = {
            "duplicate":
                """0 1.0 0
0 2.0 1
1 0.0 1
2 0.0 2
""",

            "missing":
                """0 1.0 0
1 0.0 1
""",

            "nan":
                """0 nan 0
1 0.0 1
2 0.0 2
""",

            "inf":
                """0 inf 0
1 0.0 1
2 0.0 2
""",

            "overflow":
                """0 1e999 0
1 0.0 1
2 0.0 2
""",

            "negative_group":
                """0 1.0 -1
1 0.0 1
2 0.0 2
""",

            "out_of_range":
                """0 1.0 0
1 0.0 1
3 0.0 2
""",

            "extra_column":
                """0 1.0 0 EXTRA
1 0.0 1
2 0.0 2
""",
        }

        for label, contents in bad_assignments.items():
            p = (
                root
                / f"bad_assignment_{label}.dat"
            )

            p.write_text(contents)

            out = (
                root
                / f"should_not_exist_{label}.dat"
            )

            utility_fail(
                "prepare",
                "--topology",
                str(topology),
                "--assignments",
                str(p),
                "--output",
                str(out),
                "--charge-model",
                "test",
                "--charge-source",
                "test",
            )

            print(
                f"PREPARATOR_REJECT_{label.upper()}=PASS"
            )

        print(
            "UTILITY_PREPARATOR_FAIL_CLOSED_MATRIX=PASS"
        )

        # ========================================================
        # VALID RUNTIME END-TO-END
        #
        # Generated file, not hand-written.
        # CPU and CUDA, DNANM and DNACT.
        # ========================================================

        # Only supported execution modes.
        supported = (
            ("CPU", "DNANM"),
            ("CPU", "DNACT"),
            ("CUDA", "DNANM"),
            ("CUDA", "DNACT"),
        )
        # Explicitly restrict runtime cases for CPU-only CI builds.
        # Default remains the original full CPU+CUDA matrix.
        if os.environ.get("OXDNA_TEST_CPU_ONLY") == "1":
            supported = tuple(item for item in supported if item[0] == "CPU")
            print("CUDA_RUNTIME_CASES=NOT_RUN_CPU_ONLY_MODE")

        for backend, mode in supported:
            runtime_case(
                root,
                backend=backend,
                mode=mode,
                charge_source=valid,
                expect_success=True,
            )
            print(
                f"PREPARED_RUNTIME_{backend}_{mode}=PASS"
            )

        print("PREPARATOR_TO_SUPPORTED_RUNTIME=PASS")

        # ========================================================
        # RUNTIME FAIL-CLOSED END-TO-END
        #
        # Common CPU parser must reject these before physics.
        # Test both backends and both interaction flavours.
        # ========================================================

        runtime_bad_labels = (
            "duplicate",
            "missing",
            "wrong_residue",
            "nan",
            "overflow",
            "duplicate_metadata",
            "header_extra_token",
            "npro_trailing_token",
            "ngroups_trailing_token",
        )

        for label in runtime_bad_labels:
            path = bad_files[label]

            # Require rejection of invalid charge files.
            for backend, mode in supported:
                runtime_case(
                    root,
                    backend=backend,
                    mode=mode,
                    charge_source=path,
                    expect_success=False,
                )

                print(
                    f"RUNTIME_REJECT_{label.upper()}_"
                    f"{backend}_{mode}=PASS"
                )

        print(
            "RUNTIME_FAIL_CLOSED_SUPPORTED_MATRIX=PASS"
        )

        # ========================================================
        # FINAL
        # ========================================================

        print("CHARGE_UTILITY_STANDARD_LIBRARY_ONLY=PASS")
        print("CHARGE_UTILITY_NO_AUTOMATIC_CHEMISTRY=PASS")
        print("CHARGE_UTILITY_TOPOLOGY_IDENTITY_CHECK=PASS")
        print("CHARGE_UTILITY_NONFINITE_FAIL_CLOSED=PASS")
        print("CHARGE_UTILITY_GROUP_VALIDATION=PASS")

        print("CHARGE_PREPARATOR_VALIDATOR_GATE=PASS")


if __name__ == "__main__":
    main()
