# Building and checking the source

The original oxDNA build structure is preserved. Build a CPU executable (tested on Linux with GCC 14.2) from the source root:

```bash
cmake -S . -B build -DCUDA=OFF -DDOUBLE=ON -DNATIVE_COMPILATION=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --target oxDNA --parallel 2
```

Run targeted CPU regressions in the tested environment:

```bash
export OXDNA_BIN="$PWD/build/bin/oxDNA" OXDNA_REPO="$PWD"
T=test/DNANM/protein_electrostatics
python3 -B "$T/test_source_contract.py"
python3 -B "$T/test_analytical_physics_core.py"
python3 -B "$T/test_pairenergy_term_registration.py"
python3 -B "$T/test_stress_tensor.py"
python3 -B "$T/test_dnact_cpu_nonzero_coexistence.py"
python3 -B "$T/test_surface_contract.py"
```

Additional CPU-only parser/negative tests are executed by the bundled GitHub CPU workflow. The CUDA device tests must be run on a configured NVIDIA GPU host; they **cannot** be represented by the host mock alone. Use documented tested settings in `docs/SUPPORTED_CONFIGURATIONS.md`.

The standalone example `examples/DNANM/protein_electrostatics` is synthetic and is not a standard physical protein-charge parameterization.
