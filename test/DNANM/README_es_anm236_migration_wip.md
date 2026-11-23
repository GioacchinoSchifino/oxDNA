# WIP: screened protein electrostatics on the corrected ANM PR #236 branch

Date: 2026-11-23. **Not validated or merged.** No upstream PR has been changed.

## Provenance
- Exact parent: `aa26e3236f6339e6a6429e327df35703ba2a391c` (ANM fixes, GH200 CPU/smoke validated).
- Electrostatics implementation adapted from `fe91111080ffed4bdafc270f845c0b5b2acc757c` (previous CPU-first migration).
- Upstream PR: https://github.com/lorenzo-rovigatti/oxDNA/pull/236
- Original electrostatics PR: https://github.com/lorenzo-rovigatti/oxDNA/pull/229

## Deliberate scope
- CPU-only protein–DNA and protein–protein screened contributions, explicitly opt-in.
- `protein_electrostatics` single switch; existing ANM/DNA2 interactions remain active.
- The existing PR236 bugfixes for spring identity, pair endpoint ordering, topology tokens, numeric parsing, and sparse term IDs are preserved by basing on `aa26e323`.
- CUDA ES-ON is explicitly rejected before unsupported kernels run; CUDA ES-OFF remains enabled.
- Original migrated input checks retained; positive/finite validation is now applied to explicitly set inherited screening parameters (`salt_concentration`, `dh_strength`, `dh_lambda`, `debye_huckel_rhigh`) **only when ES-ON**.
- The migrated physics test fixtures and additional invalid-DH test have been copied.
- No new workflow YAML is added.

## Critical validation gates (NOT YET PASSED for this WIP head)
1. Fresh CPU Release `-ffast-math` build and explicit `OXDNA_BIN=build/bin/oxDNA python3 -m unittest discover -s test/DNANM/protein_electrostatics -p 'test_*.py' -v`.
2. Upstream `test_run` and `test_quick`; four PR236 focused regressions; ES-OFF bitwise comparisons with the exact parent on representative same-seed fixtures and per-term observables.
3. Re-check selected-pair `PairEnergy` with non-consecutive term keys 0–10, 12, 13; test energy conservation in NVE, forces and DNA torque with finite differences and cutoff continuity.
4. Compile CUDA mixed/double and explicitly verify fail-closed ES-ON. After a successful CPU phase, repeat established full CUDA regressions on the candidate as justified.
5. If eventual NPT/pressure, MC or GPU ES-ON support is sought, separate physical and algorithmic validation is required.

## Known unresolved (do not conceal)
- The PQR numeric-chain-name parsing defect in the separate OAT candidate `40d8eb0a` (M3) is **not** fixed by this C++ patch. Keep the OAT implementation separate until corrected and tested.
- The inherited ANM custom pressure/stress observable needs independent investigation before NPT claims.
- Additional edge-list and special/circular ANM topology cases require targeted verification.
- The old `protein_dna_electrostatics` and `protein_protein_electrostatics` booleans intentionally fail with a migration message; input compatibility must be reviewed before an upstream PR.
- No compilation, CPU test, CUDA physical test or upstream maintainer approval has yet been claimed for the assembled branch.
