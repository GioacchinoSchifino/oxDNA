#!/usr/bin/env python3

"""Runtime checks for host-number -> CUDA c_number narrowing.

These tests specifically exercise a CUDA build in which the host
`number` type is double while CUDA `c_number` is float.

The test is import-safe and skips unless explicitly enabled with
OXDNA_EXPECT_CUDA_FLOAT=1.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ENABLED = (
    os.environ.get("OXDNA_EXPECT_CUDA_FLOAT")
    == "1"
)


def set_key(
    text: str,
    key: str,
    value: str,
) -> str:
    pattern = re.compile(
        rf"^(\s*{re.escape(key)}\s*=\s*).*$",
        re.MULTILINE,
    )

    if pattern.search(text):
        return pattern.sub(
            lambda m: m.group(1) + value,
            text,
            count=1,
        )

    if not text.endswith("\n"):
        text += "\n"

    return text + f"{key} = {value}\n"


def remove_key(
    text: str,
    key: str,
) -> str:
    return re.sub(
        rf"^\s*{re.escape(key)}\s*=.*$\n?",
        "",
        text,
        flags=re.MULTILINE,
    )


def read_key(
    text: str,
    key: str,
) -> str:
    pattern = re.compile(
        rf"^\s*{re.escape(key)}\s*=\s*(.*?)\s*$",
        re.MULTILINE,
    )

    match = pattern.search(text)

    if not match:
        raise RuntimeError(
            f"Missing input key {key}"
        )

    return (
        match.group(1)
        .split("#", 1)[0]
        .strip()
    )


def protein_rows(
    topology: Path,
):
    lines = [
        line.strip()
        for line in topology.read_text().splitlines()
        if line.strip()
    ]

    header = lines[0].split()

    if len(header) < 5:
        raise RuntimeError(
            f"Unexpected mixed topology header: {lines[0]}"
        )

    npro = int(header[3])

    rows = []

    for line in lines[1:]:
        fields = line.split()

        if len(fields) < 2:
            continue

        try:
            strand = int(fields[0])
        except ValueError:
            continue

        if strand < 0:
            residue = fields[1].upper()

            if len(residue) != 1:
                raise RuntimeError(
                    "Expected one-letter protein residue"
                )

            rows.append(
                (strand, residue)
            )

    if len(rows) != npro:
        raise RuntimeError(
            f"Protein count mismatch: "
            f"rows={len(rows)} npro={npro}"
        )

    return rows


def write_charge_file(
    topology: Path,
    output: Path,
    charge: str,
):
    rows = protein_rows(topology)

    lines = [
        "# DNANM_ES_CHARGES 1",
        "# index_space = protein_local",
        "# index_base = 0",
        "# charge_model = cuda-range-test",
        "# charge_source = explicit-test",
        f"# n_protein_beads = {len(rows)}",
        "# n_groups = 1",
    ]

    for i, (
        strand,
        residue,
    ) in enumerate(rows):
        lines.append(
            f"{i} {strand} {residue} "
            f"{charge} 0"
        )

    output.write_text(
        "\n".join(lines) + "\n"
    )


@unittest.skipUnless(
    ENABLED,
    "requires OXDNA_EXPECT_CUDA_FLOAT=1",
)
class CUDANumericRepresentability(
    unittest.TestCase
):
    @classmethod
    def setUpClass(cls):
        cls.repo = Path(
            os.environ["OXDNA_REPO"]
        ).resolve()

        cls.binary = Path(
            os.environ["OXDNA_BIN"]
        ).resolve()

        cls.example = (
            cls.repo
            / "examples"
            / "DNANM"
            / "KDPG"
        )

        if not cls.binary.is_file():
            raise RuntimeError(
                f"Missing oxDNA binary {cls.binary}"
            )

    def run_bad_case(
        self,
        *,
        charge: str,
        charge_scale: str,
    ):
        with tempfile.TemporaryDirectory(
            prefix="oxdna_cuda_range_"
        ) as tmp:
            work = Path(tmp)

            shutil.copytree(
                self.example,
                work,
                dirs_exist_ok=True,
            )

            source_input = (
                work
                / "input_cuda_1s"
            )

            text = source_input.read_text()

            topology_name = read_key(
                text,
                "topology",
            )

            topology = (
                work
                / topology_name
            )

            write_charge_file(
                topology,
                work / "charges_range.dat",
                charge,
            )

            for key in [
                "protein_dna_electrostatics",
                "protein_protein_electrostatics",
                "protein_charge_file",
                "protein_charge_scale",
                "protein_dna_dh_scale",
                "protein_protein_dh_scale",
                "dna_effective_charge_e",
                "exclude_intragroup_electrostatics",
            ]:
                text = remove_key(
                    text,
                    key,
                )

            text = set_key(
                text,
                "steps",
                "1",
            )

            text += (
                "\n"
                "protein_dna_electrostatics = false\n"
                "protein_protein_electrostatics = true\n"
                "protein_charge_file = charges_range.dat\n"
                f"protein_charge_scale = {charge_scale}\n"
                "protein_dna_dh_scale = 1.0\n"
                "protein_protein_dh_scale = 1.0\n"
                "dna_effective_charge_e = 0.815\n"
                "exclude_intragroup_electrostatics = false\n"
            )

            inp = (
                work
                / "input_cuda_range"
            )

            inp.write_text(text)

            proc = subprocess.run(
                [
                    str(self.binary),
                    str(inp),
                ],
                cwd=work,
                capture_output=True,
                text=True,
            )

            log = ""

            log_file = (
                work
                / "log.dat"
            )

            if log_file.is_file():
                log = log_file.read_text(
                    errors="replace"
                )

            combined = (
                proc.stdout
                + "\n"
                + proc.stderr
                + "\n"
                + log
            )

            self.assertNotEqual(
                proc.returncode,
                0,
                msg=(
                    "CUDA runtime unexpectedly accepted "
                    "a value that cannot be represented "
                    "as c_number"
                ),
            )

            # A valid early host/device range guard may
            # reject before the explicit c_number cast.
            self.assertIn(
                "Running on device 0",
                combined,
            )
            self.assertIn(
                "CUDA backend precision: mixed",
                combined,
            )
            self.assertIn(
                "Loaded 605 protein charges",
                combined,
            )
            self.assertNotIn(
                "DNACT CUDA angular ANM is not implemented",
                combined,
            )
            self.assertTrue(
                (
                    "CUDA reduced protein charge out of range at particle"
                    in combined
                ) or (
                    "not representable as CUDA c_number" in combined
                    and "reduced protein charge" in combined
                ),
                msg=(
                    "CUDA did not reject the oversized reduced "
                    "protein charge for the expected reason.\n"
                    + combined
                ),
            )

    def test_charge_scale_outside_cuda_float_range(
        self,
    ):
        self.run_bad_case(
            charge="1.0",
            charge_scale="1e100",
        )

    def test_charge_outside_cuda_float_range(
        self,
    ):
        self.run_bad_case(
            charge="1e100",
            charge_scale="1.0",
        )


if __name__ == "__main__":
    unittest.main()
