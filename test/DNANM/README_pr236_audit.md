# Focused ANM regression tests for PR #236

These Python scripts are **standalone tests, not GitHub Actions workflows**. They verify regressions in the ANM refactoring without adding CI jobs. They require a CPU oxDNA binary built from the tested checkout and the repository's existing KDPG example.

From the repository root:

```bash
cmake -S . -B build -DCUDA=OFF
cmake --build build -j2
bash test/DNANM/run_pr236_cpu_regressions.sh build/bin/oxDNA ./pr236_focused_results
```

The single fail-fast runner executes all four tests and **exits nonzero if any one fails**, even with the Release `-ffast-math` compiler flag. The pair-observable test also writes JSON for inspection; the optional `--compare` mode compares two independently generated snapshots. A failed subprocess or missing output also causes a failure, not a silent PASS.

What these tests cover:

- **Bonded ANMT energy accounting**: total and split terms agree for a perturbed protein; cross-chain springs connecting consecutively numbered but distinct chains do not receive artificial bending/torsion.
- **PairForce**: reversed bonded pair calls retain a nonzero angular torque.
- **PairEnergy**: data column count matches the header for ANM systems.
- **New topologies**: mixed DNANM accepts numeric DNA base tokens `(0)` as the standard DNA2 model does.
- **ANMT parameter parsing**: `nan`, `inf`, incomplete decimals, and overflow are rejected in cosine, spring and global stiffness fields, while valid exponent/signed decimals remain accepted.

The CPU regressions **do not certify CUDA**. In particular, the CUDA parser change needs a CUDA build and CPU/GPU regression checks on a GPU node. Use Erik's external suite: https://github.com/ErikPoppleton/oxDNA_regression_tests. The new `CUDA_list=edge` code path also merits separate checking. The pressure/stress-tensor issue is independent and intentionally outside these fixes.
