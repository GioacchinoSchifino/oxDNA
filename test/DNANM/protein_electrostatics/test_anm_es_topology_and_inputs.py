#!/usr/bin/env python3
"""Regression cases specific to the 2026 ANM refactor and unified ES switch."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import test_analytical_physics_core as physics

OXDNA = Path(os.environ["OXDNA_BIN"]).resolve()

NEW_DNA_FIRST = """4 4 5->3
A type=DNA
K type=peptide
K type=peptide
K type=peptide
"""

NEW_PROTEIN_FIRST = """4 4 5->3
K type=peptide
K type=peptide
K type=peptide
A type=DNA
"""


class ANMRefactorTopologyAndValidation(unittest.TestCase):
    def _run_raw(self, root, name, topology=physics.TOP, charges=None,
                 input_override=None, protein_first=False):
        case = root / name
        case.mkdir()
        (case / "system.top").write_text(topology)
        if protein_first:
            rows = physics.conf_text().splitlines(keepends=True)
            conf = "".join(rows[:3] + rows[4:7] + [rows[3]])
        else:
            conf = physics.conf_text()
        (case / "init.dat").write_text(conf)
        if charges is None:
            charges = physics.charges_text((1., 0., 0.))
        (case / "protein_charges.dat").write_text(charges)
        inp = physics.input_text("DNANM", True, False,
                                  (0, 3) if protein_first else (0, 1))
        if input_override:
            inp = input_override(inp)
        (case / "input").write_text(inp)
        proc = subprocess.run([str(OXDNA), "input"], cwd=case,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, timeout=60)
        log = (case / "log.dat").read_text(errors="replace") if (case / "log.dat").exists() else ""
        return proc, log, case

    def test_01_new_style_dna_first_matches_old(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = physics.run_case(root, "legacy", pd=True,
                pair=(0, 1), charges=(1., 0., 0.))
            with patch.object(physics, "TOP", NEW_DNA_FIRST):
                modern = physics.run_case(root, "new", pd=True,
                    pair=(0, 1), charges=(1., 0., 0.))
            self.assertAlmostEqual(baseline["energy"], modern["energy"], delta=2e-6)
            for pid in (0, 1):
                for key in ("force", "torque"):
                    for x, y in zip(baseline["force_info"][pid][key],
                                    modern["force_info"][pid][key]):
                        self.assertAlmostEqual(x, y, delta=2e-6)

    def test_02_new_style_protein_first_maps_global_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref = physics.run_case(root, "legacy", pd=True,
                pair=(0, 1), charges=(1., 0., 0.))
            proc, log, case = self._run_raw(
                root, "protein_first", topology=NEW_PROTEIN_FIRST, protein_first=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stdout + "\n" + log)
            modern_e = physics.first_block_pair_energy(case / "pair_energy.dat", (0, 3))
            self.assertAlmostEqual(ref["energy"], modern_e, delta=2e-6)

    def test_03_invalid_residue_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            charges = physics.charges_text((1., 0., 0.)).replace(
                "0 -1 K ", "0 -1 R ", 1)
            proc, log, _ = self._run_raw(Path(tmp), "bad_residue", charges=charges)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("mismatch", proc.stdout.lower() + log.lower())

    def test_04_nonfinite_charge_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            charges = physics.charges_text((1., 0., 0.)).replace(
                "1.000000000000", "nan", 1)
            proc, log, _ = self._run_raw(Path(tmp), "nan_charge", charges=charges)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("numeric", proc.stdout.lower() + log.lower())

    def test_05_cuda_es_on_fails_closed_without_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc, log, _ = self._run_raw(Path(tmp), "cuda_disallowed",
                input_override=lambda s: s.replace("backend = CPU", "backend = CUDA"))
            self.assertNotEqual(proc.returncode, 0)
            combined = (proc.stdout + log).lower()
            self.assertIn("cpu-only", combined)

    def test_06_deprecated_separate_switch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc, log, _ = self._run_raw(Path(tmp), "old_switch",
                input_override=lambda s: s + "\nprotein_protein_electrostatics = true\n")
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("single switch", (proc.stdout + log).lower())

    def test_07_disabled_and_zero_charges_preserve_pair_energy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = physics.run_case(root, "off", pd=False, pp=False,
                pair=(0, 1), charges=(0., 0., 0.))
            candidate = physics.run_case(root, "on_zero", pd=True, pp=True,
                pair=(0, 1), charges=(0., 0., 0.))
            self.assertAlmostEqual(baseline["energy"], candidate["energy"], delta=1e-8)

    def test_08_charge_file_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proc, log, case = self._run_raw(root, "missing_charge",
                input_override=lambda s: s.replace(
                    "protein_charge_file = protein_charges.dat",
                    "protein_charge_file = missing_charges.txt"))
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("charge", (proc.stdout + log).lower())


if __name__ == "__main__":
    unittest.main()
