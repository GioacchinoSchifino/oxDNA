# DNANM protein electrostatics example

This directory provides a minimal synthetic example of the optional protein electrostatic interaction for DNANM.

The system contains one DNA nucleotide and three protein beads. Two protein beads carry explicit positive charges. Their non-bonded protein--protein electrostatic interaction is enabled in the input file.

Run the example from this directory with:

    /path/to/oxDNA input

Protein electrostatics is opt-in. Existing DNANM/DNACT behaviour is retained when both protein electrostatic switches are false.

The charges in this example are illustrative values chosen to make the optional interaction easy to exercise. They are not intended as a transferable physical protein-charge parametrisation.

For real systems, provide explicit charge assignments and use utils/dnanm_protein_charges.py to prepare and validate the version-1 charge file. See docs/source/input.md and docs/source/utils.md for the full format and options.
