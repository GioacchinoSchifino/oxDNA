# Scientific model, assumptions, and interpretation

`DNANM` couples the nucleotide oxDNA2 model to an ANM protein representation. `DNACT` adds the established protein angular model. The electrostatic extension is enabled by runtime switches and uses screened effective (Debye–Hückel-type) interactions for protein–DNA and optional interprotein bead pairs. The protein–DNA site is nucleotide BACK, so position and torque must be considered together.

These terms are **sequence-nonspecific**: they do not resolve hydrogen-bond specificity, ions explicitly, sequence-dependent transcription-factor binding, pKa, charge regulation, polarization or atomistic solvation. Salt enters via the existing oxDNA2 screening conventions; mapping real-solution parameters onto a CG potential remains a modeling decision.

Protein charge files are required when enabling relevant terms. Users must define charge assignment/protonation and document source, pH, terminal convention, ionic strength, and any coarse-graining assumptions. The historical S20 file cannot be presented as a validated general parameter set.

**Supported computational scope** is narrower than the code's full set of possible switches; see `SUPPORTED_CONFIGURATIONS.md`. One-step force/stress parity tests confirm implemented algebra in their fixtures, not population distributions or experimental physical validation.
