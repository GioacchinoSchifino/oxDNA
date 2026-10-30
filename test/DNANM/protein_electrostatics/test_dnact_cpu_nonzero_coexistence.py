#!/usr/bin/env python3

"""Regression test for CPU DNACT angular/electrostatic coexistence."""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


def set_key(text, key, value):
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


def remove_key(text, key):
    return re.sub(
        rf"^\s*{re.escape(key)}\s*=.*$\n?",
        "",
        text,
        flags=re.MULTILINE,
    )


def read_key(text, key):
    pattern = re.compile(
        rf"^\s*{re.escape(key)}\s*=\s*(.*?)\s*$",
        re.MULTILINE,
    )

    match = pattern.search(text)

    if match is None:
        raise RuntimeError(f"Missing input key {key}")

    return match.group(1).split("#", 1)[0].strip()


def get_protein_rows(topology):
    lines = [
        line.strip()
        for line in topology.read_text().splitlines()
        if line.strip()
    ]

    header = lines[0].split()

    if len(header) < 5:
        raise RuntimeError(
            f"Unexpected topology header: {lines[0]}"
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
                    f"Expected one-letter residue, got {residue}"
                )

            rows.append((strand, residue))

    if len(rows) != npro:
        raise RuntimeError(
            f"Protein count mismatch: rows={len(rows)} npro={npro}"
        )

    return rows


def write_charges(topology, output):
    rows = get_protein_rows(topology)

    lines = [
        "# DNANM_ES_CHARGES 1",
        "# index_space = protein_local",
        "# index_base = 0",
        "# charge_model = dnact-coexistence-test",
        "# charge_source = synthetic-explicit-test",
        f"# n_protein_beads = {len(rows)}",
        "# n_groups = 1",
    ]

    for index, (strand, residue) in enumerate(rows):
        lines.append(
            f"{index} {strand} {residue} 1.0 0"
        )

    output.write_text(
        "\n".join(lines) + "\n"
    )


def particle_state(path):
    lines = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip()
    ]

    if len(lines) < 4:
        raise RuntimeError(
            f"Malformed configuration: {path}"
        )

    rows = []

    # t=, b= and E= are metadata; only particle rows are compared.
    for line in lines[3:]:
        try:
            values = tuple(
                float(x)
                for x in line.split()
            )
        except ValueError:
            continue

        if not values:
            continue

        if not all(math.isfinite(x) for x in values):
            raise RuntimeError(
                f"Non-finite particle state: {path}"
            )

        rows.append(values)

    if not rows:
        raise RuntimeError(
            f"No particle state in {path}"
        )

    return tuple(rows)


def max_delta(a, b):
    if len(a) != len(b):
        raise RuntimeError(
            f"State length mismatch: {len(a)} != {len(b)}"
        )

    return max(
        abs(x - y)
        for row_a, row_b in zip(a, b)
        for x, y in zip(row_a, row_b)
    )


class DNACTCPUNonzeroCoexistence(unittest.TestCase):

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

        source = (
            cls.example
            / "input_dnact"
        ).read_text()

        # Lock the regression to the non-zero values shipped by
        # the upstream KDPG DNACT example.
        if read_key(source, "bending_k") != "50.0":
            raise RuntimeError(
                "Unexpected upstream KDPG bending_k"
            )

        if read_key(source, "torsion_k") != "50.0":
            raise RuntimeError(
                "Unexpected upstream KDPG torsion_k"
            )

    def run_case(
        self,
        root,
        label,
        angular_nonzero,
        electrostatics,
    ):
        work = root / label

        shutil.copytree(
            self.example,
            work,
        )

        source_input = work / "input_dnact"
        text = source_input.read_text()

        topology_name = read_key(
            text,
            "topology",
        )

        topology = work / topology_name

        text = set_key(text, "steps", "1")
        text = set_key(text, "thermostat", "no")
        text = set_key(text, "print_conf_interval", "1")
        text = set_key(text, "print_energy_every", "0")
        text = set_key(
            text,
            "trajectory_file",
            "trajectory_test.dat",
        )
        text = set_key(
            text,
            "lastconf_file",
            "last_test.dat",
        )

        if angular_nonzero:
            text = set_key(
                text,
                "bending_k",
                "50.0",
            )
            text = set_key(
                text,
                "torsion_k",
                "50.0",
            )
        else:
            text = set_key(
                text,
                "bending_k",
                "0.0",
            )
            text = set_key(
                text,
                "torsion_k",
                "0.0",
            )

        feature_keys = [
            "protein_dna_electrostatics",
            "protein_protein_electrostatics",
            "protein_charge_file",
            "protein_charge_scale",
            "protein_dna_dh_scale",
            "protein_protein_dh_scale",
            "dna_effective_charge_e",
            "exclude_intragroup_electrostatics",
        ]

        for key in feature_keys:
            text = remove_key(
                text,
                key,
            )

        if electrostatics:
            write_charges(
                topology,
                work / "charges_test.dat",
            )

            text += (
                "\n"
                "protein_dna_electrostatics = false\n"
                "protein_protein_electrostatics = true\n"
                "protein_charge_file = charges_test.dat\n"
                "protein_charge_scale = 1.0\n"
                "protein_dna_dh_scale = 1.0\n"
                "protein_protein_dh_scale = 10.0\n"
                "dna_effective_charge_e = 0.815\n"
                "exclude_intragroup_electrostatics = false\n"
            )

        inp = work / "input_test"
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

        if proc.returncode != 0:
            log = work / "log.dat"

            raise RuntimeError(
                f"{label} failed rc={proc.returncode}\n"
                f"{proc.stdout}\n"
                f"{proc.stderr}\n"
                f"{log.read_text(errors='replace') if log.exists() else ''}"
            )

        last = work / "last_test.dat"

        if not last.is_file():
            raise RuntimeError(
                f"{label}: last configuration missing"
            )

        return particle_state(last)

    def test_nonzero_angular_terms_and_es_coexist(self):

        with tempfile.TemporaryDirectory(
            prefix="oxdna_dnact_coexist_"
        ) as tmp:
            root = Path(tmp)

            off_nonzero = self.run_case(
                root,
                "off_nonzero",
                True,
                False,
            )

            off_zero = self.run_case(
                root,
                "off_zero",
                False,
                False,
            )

            on_nonzero = self.run_case(
                root,
                "on_nonzero",
                True,
                True,
            )

            on_zero = self.run_case(
                root,
                "on_zero",
                False,
                True,
            )

            angular_off = max_delta(
                off_nonzero,
                off_zero,
            )

            angular_on = max_delta(
                on_nonzero,
                on_zero,
            )

            es_nonzero = max_delta(
                on_nonzero,
                off_nonzero,
            )

            es_zero = max_delta(
                on_zero,
                off_zero,
            )

            print(
                "DNACT_CPU_ANGULAR_OFF_DELTA="
                f"{angular_off:.17e}"
            )

            print(
                "DNACT_CPU_ANGULAR_ON_DELTA="
                f"{angular_on:.17e}"
            )

            print(
                "DNACT_CPU_ES_NONZERO_DELTA="
                f"{es_nonzero:.17e}"
            )

            print(
                "DNACT_CPU_ES_ZERO_DELTA="
                f"{es_zero:.17e}"
            )

            self.assertGreater(
                angular_off,
                1.0e-5,
            )

            self.assertGreater(
                angular_on,
                1.0e-5,
            )

            self.assertGreater(
                es_nonzero,
                1.0e-6,
            )

            self.assertGreater(
                es_zero,
                1.0e-6,
            )

            ratio = angular_on / angular_off

            print(
                "DNACT_CPU_ANGULAR_RATIO="
                f"{ratio:.17e}"
            )

            self.assertGreater(
                ratio,
                0.90,
            )

            self.assertLess(
                ratio,
                1.10,
            )


if __name__ == "__main__":
    unittest.main()
