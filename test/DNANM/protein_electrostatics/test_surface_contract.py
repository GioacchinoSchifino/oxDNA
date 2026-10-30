#!/usr/bin/env python3

import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
KDPG = REPO / "examples" / "DNANM" / "KDPG"


def replace_setting(text: str, key: str, value: str) -> str:
    pattern = re.compile(
        rf"(?m)^[ \t]*{re.escape(key)}[ \t]*=.*$"
    )
    replacement = f"{key} = {value}"
    if pattern.search(text):
        return pattern.sub(replacement, text)
    return text.rstrip() + "\n" + replacement + "\n"


class ProteinElectrostaticsSurfaceContract(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        raw = os.environ.get("OXDNA_BIN", "")
        cls.oxdna = Path(raw).resolve()

        if not cls.oxdna.is_file():
            raise RuntimeError(
                f"OXDNA_BIN does not point to an executable file: {cls.oxdna}"
            )

        if not os.access(cls.oxdna, os.X_OK):
            raise RuntimeError(
                f"OXDNA_BIN is not executable: {cls.oxdna}"
            )

        required = (
            "input_dnanm",
            "input_dnact",
            "generated.top",
            "s1.dat",
            "kdpg_anm.par",
            "kdpg_act.par",
        )

        missing = [name for name in required if not (KDPG / name).exists()]
        if missing:
            raise RuntimeError(
                "Missing official upstream KDPG fixture(s): "
                + ", ".join(missing)
            )

    def run_case(self, input_name: str, switch_name: str):
        with tempfile.TemporaryDirectory(
            prefix="oxdna_pdpp_surface_"
        ) as tmp_raw:

            tmp = Path(tmp_raw)
            case = tmp / "KDPG"
            shutil.copytree(KDPG, case)

            inp = case / input_name
            text = inp.read_text()

            # Keep the runtime tiny and deterministic.
            text = replace_setting(text, "steps", "1")
            text = replace_setting(text, "seed", "17")
            text = replace_setting(text, "print_energy_every", "1")
            text = replace_setting(text, "print_conf_interval", "1")

            # Explicitly enable exactly one new feature.
            text += f"\n{switch_name} = true\n"

            # Deliberately DO NOT provide protein_charge_file.
            inp.write_text(text)

            proc = subprocess.run(
                [str(self.oxdna), inp.name],
                cwd=case,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=60,
            )

            log_text = ""

            logfile = case / "log.dat"

            if logfile.exists():
                log_text = logfile.read_text(
                    errors="replace"
                )

            combined_output = (
                proc.stdout
                + "\n"
                + log_text
            )

            return proc.returncode, combined_output

    def assert_missing_charge_file_rejected(
        self,
        input_name: str,
        switch_name: str,
    ):
        rc, output = self.run_case(input_name, switch_name)

        self.assertNotEqual(
            rc,
            0,
            msg=(
                f"{input_name} accepted {switch_name}=true "
                "without protein_charge_file.\n"
                "Expected fail-closed behaviour.\n\n"
                f"OUTPUT:\n{output}"
            ),
        )

        self.assertRegex(
            output,
            r"protein_charge_file.*required",
            msg=(
                "The run failed, but not for the required "
                "protein_charge_file contract.\n\n"
                f"OUTPUT:\n{output}"
            ),
        )

    def test_dnanm_pd_requires_charge_file(self):
        self.assert_missing_charge_file_rejected(
            "input_dnanm",
            "protein_dna_electrostatics",
        )

    def test_dnanm_pp_requires_charge_file(self):
        self.assert_missing_charge_file_rejected(
            "input_dnanm",
            "protein_protein_electrostatics",
        )

    def test_dnact_pd_requires_charge_file(self):
        self.assert_missing_charge_file_rejected(
            "input_dnact",
            "protein_dna_electrostatics",
        )

    def test_dnact_pp_requires_charge_file(self):
        self.assert_missing_charge_file_rejected(
            "input_dnact",
            "protein_protein_electrostatics",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
