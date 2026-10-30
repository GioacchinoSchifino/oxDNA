#!/usr/bin/env python3

import unittest
from pathlib import Path


HERE = Path(__file__).resolve()
REPO = HERE.parents[3]

HEADER = REPO / "src" / "Interactions" / "DNANMInteraction.h"
SOURCE = REPO / "src" / "Interactions" / "DNANMInteraction.cpp"


class ProteinElectrostaticsSourceContract(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.header = HEADER.read_text()
        cls.source = SOURCE.read_text()

    def test_public_term_ids_exist(self):
        self.assertIn("PRO_DNA_DH", self.header)
        self.assertIn("PRO_PRO_DH", self.header)

    def test_shared_feature_switches_exist(self):
        self.assertIn("_protein_dna_electrostatics", self.header)
        self.assertIn("_protein_protein_electrostatics", self.header)

    def test_new_particle_aware_stress_api_is_used(self):
        # The legacy fork-specific capability method must never return.
        self.assertNotIn("supports_stress_tensor", self.header)
        self.assertNotIn("supports_stress_tensor", self.source)

        # The implementation must ultimately use particle-aware stress updates.
        self.assertRegex(
            self.source,
            r"_update_stress_tensor\s*\(\s*[^,\n]+\s*,\s*"
            r"[^,\n]+\s*,\s*[^,\n]+\s*,",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
