# PR #236 — H100 CUDA validation

24 November 2026

The archive contains my validation note, the case-specific CLN025 sampling patch, the saved numerical results, diagnostic scripts, and reproduction instructions. It covers the ANM refactoring only; it does not include the protein electrostatics implementation.

- Tested ANM fixes: `aa26e3236f6339e6a6429e327df35703ba2a391c`
- Upstream refactoring: `81dafaa3a4c93da656904f4334db5e3bc54e9351`
- Regression suite: `a1da3dfc74d78cc2feb9f48a0a284f6744960d7e`

[Download the validation archive](./oxDNA_PR236_H100_CUDA_Validation_20261124.zip)

SHA-256: `c53edb506e179d2835af9cf65c4657e9413d719322373cfc5d82cb4dceebf950`

After extracting the ZIP, run `python3 verify_results.py` in the extracted directory to check the saved reports. See `REPRODUCE.md` for rerunning the tests. The offline verifier does not reconstruct raw trajectories or device arrays.
