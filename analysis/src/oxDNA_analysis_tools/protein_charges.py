import argparse
import os
import sys
import math
from oxDNA_analysis_tools.UTILS.RyeReader import strand_describe
from oxDNA_analysis_tools.UTILS.logger import log, logger_settings
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

# Charge-file format version understood by the DNANM/DNACT interaction
# (`protein_charge_file`).
CHARGE_FILE_VERSION = 1

# Simple sequence-based charges at roughly neutral pH. Histidine is treated as
# neutral by default (it is only partially protonated at pH 7) and can be
# changed with --his-charge.
SIMPLE_CHARGES = {'K': 1.0, 'R': 1.0, 'D': -1.0, 'E': -1.0}

# Three-letter to one-letter residue codes, including the protonation-state
# variants written by PDB2PQR / AMBER / CHARMM / GROMACS tools.
THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
    'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
    'TYR': 'Y', 'VAL': 'V',
    # protonation-state variants
    'HID': 'H', 'HIE': 'H', 'HIP': 'H', 'HSD': 'H', 'HSE': 'H', 'HSP': 'H',
    'ASH': 'D', 'GLH': 'E', 'LYN': 'K', 'CYX': 'C', 'CYM': 'C', 'CYSH': 'C',
    'ARN': 'R', 'TYM': 'Y',
}


class ChargeError(Exception):
    """Raised for any inconsistent or unsupported input (the tool never guesses)."""


def read_protein_topology(topology: str) -> List[Tuple[int, str]]:
    """
    Return (canonical peptide strand ID, one-letter residue) per protein bead.
    Reuses OAT's standard topology reader for both legacy and 5->3 files.

    Legacy peptide IDs are the negative numbers in the file; new-style
    peptide strands use -1, -2, ... in peptide appearance order to match
    the CPU charge-file parser in DNANMInteraction.
    """
    with open(topology) as handle:
        header = handle.readline().split()
    if not header:
        raise ChargeError(f"Topology '{topology}' is empty")
    try:
        n_total = int(header[0])
    except (ValueError, IndexError):
        raise ChargeError(f"Invalid topology header in '{topology}'")
    is_new = header[-1] == "5->3"
    if not is_new and len(header) != 5:
        raise ChargeError("Legacy mixed ANM topology requires five integer header fields")

    try:
        system, monomers = strand_describe(topology)
    except (IndexError, ValueError, RuntimeError) as exc:
        raise ChargeError(f"Failed to read ANM topology '{topology}': {exc}") from exc

    if len(monomers) != n_total:
        raise ChargeError(f"Topology declares {n_total} particles but describes {len(monomers)}")
    proteins: List[Tuple[int, str]] = []
    peptide_ordinal = 0
    for strand in system.strands:
        if not strand.is_peptide():
            continue
        peptide_ordinal += 1
        label = -peptide_ordinal if is_new else strand.id
        for bead in strand:
            code = str(bead.btype).upper()
            if len(code) != 1 or not ('A' <= code <= 'Z'):
                raise ChargeError(f"Invalid protein residue '{bead.btype}' on peptide {label}")
            proteins.append((label, code))

    if not proteins:
        raise ChargeError("The topology contains no protein beads")
    if not is_new:
        try:
            expected_protein = int(header[3])
        except ValueError as exc:
            raise ChargeError("Invalid legacy topology protein count") from exc
        if expected_protein != len(proteins):
            raise ChargeError(
                f"Topology declares {expected_protein} protein beads but found {len(proteins)}")
    return proteins

def simple_charges(residues: Sequence[Tuple[int, str]], his_charge: float = 0.0,
                   termini: bool = False) -> List[float]:
    """
        Assign sequence-based charges: Lys/Arg = +1, Asp/Glu = -1, His = `his_charge`, others = 0.

        Parameters:
            residues (Sequence[Tuple[int, str]]): (strand id, residue code) for each protein bead.
            his_charge (float): Charge assigned to histidine. (Default: 0.0)
            termini (bool): If True, add +1 to the first and -1 to the last bead of every strand,
                to represent the N- and C-terminal groups. (Default: False)

        Returns:
            List[float]: One charge (in units of e) per protein bead.
    """
    if not math.isfinite(his_charge):
        raise ChargeError("Histidine charge must be finite")
    table = dict(SIMPLE_CHARGES)
    table['H'] = his_charge
    charges = [table.get(res, 0.0) for _, res in residues]
    if termini:
        for i, (strand, _) in enumerate(residues):
            if i == 0 or residues[i - 1][0] != strand:
                charges[i] += 1.0
            if i == len(residues) - 1 or residues[i + 1][0] != strand:
                charges[i] -= 1.0
    return charges


def read_pqr_residue_charges(pqr: str) -> List[Tuple[str, str, str, float]]:
    """
    Read PQR ATOM records as (chain, residue ID, one-letter code, net charge).
    Chain and residue IDs are retained so multiple protein copies cannot be
    silently conflated. Noncontiguous duplicate residue identifiers are errors.
    """
    residues: "OrderedDict[tuple, List]" = OrderedDict()
    closed = set()
    previous = None
    with open(pqr) as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            fields = line.split()
            if len(fields) < 10:
                raise ChargeError(f"Cannot parse PQR line: {line.rstrip()}")
            name = fields[3].upper()
            try:
                float(fields[4])
                chain, residue_id = "", fields[4]
            except ValueError:
                chain, residue_id = fields[4], fields[5]
            try:
                charge = float(fields[-2])
            except ValueError as exc:
                raise ChargeError(f"Cannot read charge from PQR line: {line.rstrip()}") from exc
            if not math.isfinite(charge):
                raise ChargeError(f"Nonfinite PQR charge for {chain}:{residue_id}")
            if name not in THREE_TO_ONE:
                raise ChargeError(f"Unsupported residue '{name}' in PQR file")
            key = (chain, residue_id)
            if previous is not None and key != previous:
                closed.add(previous)
            if key in closed:
                raise ChargeError(f"PQR residue {chain}:{residue_id} is split across noncontiguous records")
            previous = key
            if key not in residues:
                residues[key] = [THREE_TO_ONE[name], 0.0]
            elif residues[key][0] != THREE_TO_ONE[name]:
                raise ChargeError(f"Conflicting PQR residue codes for {chain}:{residue_id}")
            residues[key][1] += charge
            if not math.isfinite(residues[key][1]):
                raise ChargeError(f"Nonfinite PQR residue net charge for {chain}:{residue_id}")

    if not residues:
        raise ChargeError(f"No ATOM records found in '{pqr}'")
    return [(chain, rid, code, charge)
            for (chain, rid), (code, charge) in residues.items()]


def pqr_charges(residues: Sequence[Tuple[int, str]], pqr: str,
                pqr_chain_order: Optional[str] = None) -> List[float]:
    """
    Match peptide strands to PQR chains by their *full sequences*.

    With duplicated sequences (e.g. identical subunits that have different
    protonation states), an explicit --pqr-chain-order is mandatory. This
    prevents assigning the right sequence's charges to the wrong subunit.
    """
    top_chains: "OrderedDict[int, List[str]]" = OrderedDict()
    for label, code in residues:
        top_chains.setdefault(label, []).append(code)
    pqr_res = read_pqr_residue_charges(pqr)
    pqr_chains: "OrderedDict[str, List[Tuple[str, float]]]" = OrderedDict()
    for chain, _rid, code, charge in pqr_res:
        pqr_chains.setdefault(chain, []).append((code, charge))

    topology_labels = list(top_chains)
    if pqr_chain_order:
        mapping = [c.strip() for c in pqr_chain_order.split(",")]
        if len(mapping) != len(topology_labels) or len(set(mapping)) != len(mapping):
            raise ChargeError("PQR chain order must list one distinct chain per peptide strand")
        if any(chain not in pqr_chains for chain in mapping):
            raise ChargeError("PQR chain order contains a chain absent from the PQR file")
    else:
        mapping = []
        used = set()
        for label in topology_labels:
            desired = top_chains[label]
            candidates = [c for c, data in pqr_chains.items()
                          if c not in used and [res for res, _q in data] == desired]
            if len(candidates) != 1:
                raise ChargeError(
                    f"Ambiguous or missing PQR chain for peptide {label}; "
                    "use --pqr-chain-order A,B,... to fix the mapping")
            mapping.append(candidates[0])
            used.add(candidates[0])

    if len(pqr_chains) != len(mapping):
        raise ChargeError("PQR has extra/missing protein chains relative to topology")

    result = []
    for label, chain in zip(topology_labels, mapping):
        expected = top_chains[label]
        observed = pqr_chains[chain]
        if [res for res, _q in observed] != expected:
            raise ChargeError(
                f"Peptide {label} does not match PQR chain '{chain}' "
                "(wrong residue order, identity or length)")
        result.extend(q for _res, q in observed)
    return result

def assign_groups(residues: Sequence[Tuple[int, str]], mode: str = 'strand',
                  group_file: Optional[str] = None) -> List[int]:
    """
        Assign a group id to each protein bead. With the default
        `exclude_intragroup_electrostatics = true`, beads in the same group do not interact
        electrostatically, so a group should be a rigid unit already held together by the ANM springs.

        Parameters:
            residues (Sequence[Tuple[int, str]]): (strand id, residue code) for each protein bead.
            mode (str): 'strand' (one group per topology strand), 'single' (one group for all beads)
                or 'residue' (every bead its own group). (Default: 'strand')
            group_file (str): Optional text file with one integer group id per protein bead.
                Overrides `mode`.

        Returns:
            List[int]: Contiguous group ids starting from 0, one per protein bead.
    """
    if group_file is not None:
        with open(group_file) as f:
            raw = [l.split('#')[0].strip() for l in f]
        try:
            ids = [int(x) for x in raw if x]
        except ValueError:
            raise ChargeError(f"Group file '{group_file}' must contain one integer per line")
        if len(ids) != len(residues):
            raise ChargeError(
                f"Group file has {len(ids)} entries but there are {len(residues)} protein beads")
    elif mode == 'strand':
        ids = [s for s, _ in residues]
    elif mode == 'single':
        ids = [0] * len(residues)
    elif mode == 'residue':
        ids = list(range(len(residues)))
    else:
        raise ChargeError(f"Unknown group mode '{mode}'")

    # renumber contiguously from 0 in order of first appearance
    remap: Dict[int, int] = {}
    return [remap.setdefault(g, len(remap)) for g in ids]


def write_charge_file(out_file: str, residues: Sequence[Tuple[int, str]], charges: Sequence[float],
                      groups: Sequence[int], model: str, source: str) -> None:
    """
        Write a version-1 DNANM/DNACT protein charge file.

        Parameters:
            out_file (str): Path of the file to write.
            residues (Sequence[Tuple[int, str]]): (strand id, residue code) for each protein bead.
            charges (Sequence[float]): Charge in units of e for each protein bead.
            groups (Sequence[int]): Group id for each protein bead.
            model (str): Short description of how the charges were obtained (written in the header).
            source (str): Description of the input data (written in the header).
    """
    n = len(residues)
    if not (len(charges) == len(groups) == n):
        raise ChargeError("residues, charges and groups must have the same length")
    # Metadata is parsed as 'key = value' per line, so keep values on one line.
    model = ' '.join(model.split())
    source = ' '.join(source.split())
    if any(not math.isfinite(float(q)) for q in charges):
        raise ChargeError("Cannot write nonfinite protein charges")
    with open(out_file, 'w') as f:
        f.write(f"# DNANM_ES_CHARGES {CHARGE_FILE_VERSION}\n")
        f.write("# index_space = protein_local\n")
        f.write("# index_base = 0\n")
        f.write(f"# n_protein_beads = {n}\n")
        f.write(f"# n_groups = {len(set(groups))}\n")
        f.write(f"# charge_model = {model}\n")
        f.write(f"# charge_source = {source}\n")
        f.write("# generator = oat protein_charges\n")
        f.write("# columns = protein_index topology_strand residue charge_e group_id\n")
        for i, ((strand, res), q, g) in enumerate(zip(residues, charges, groups)):
            f.write(f"{i} {strand} {res} {q:.6f} {g}\n")


def protein_charges(topology: str, out_file: str, pqr: Optional[str] = None, his_charge: float = 0.0,
                    termini: bool = False, group_by: str = 'strand',
                    group_file: Optional[str] = None,
                    pqr_chain_order: Optional[str] = None) -> Tuple[List[float], List[int]]:
    """
        Generate a protein charge file for the DNANM/DNACT electrostatics from a topology file.

        Parameters:
            topology (str): Path to the DNANM/DNACT topology file.
            out_file (str): Path of the charge file to write.
            pqr (str): Optional PQR file (from PDB2PQR/PROPKA). If given, its per-residue charges are
                used instead of the simple sequence-based model. (Default: None)
            his_charge (float): Histidine charge for the sequence-based model. (Default: 0.0)
            termini (bool): Add terminal charges in the sequence-based model. (Default: False)
            group_by (str): 'strand', 'single' or 'residue'. (Default: 'strand')
            group_file (str): Optional per-bead group id file. Overrides `group_by`.

        Returns:
            Tuple[List[float], List[int]]: The charge and group id of every protein bead.
    """
    residues = read_protein_topology(topology)

    if pqr is not None:
        charges = pqr_charges(residues, pqr, pqr_chain_order)
        model = "per-residue net charge from PQR (PDB2PQR/PROPKA protonation states)"
        source = f"topology={os.path.basename(topology)} pqr={os.path.basename(pqr)}"
    else:
        charges = simple_charges(residues, his_charge, termini)
        model = (f"simple sequence-based: K,R=+1 D,E=-1 H={his_charge:g}"
                 f"{' termini=+1/-1' if termini else ''} (not pKa-based)")
        source = f"topology={os.path.basename(topology)}"

    groups = assign_groups(residues, group_by, group_file)
    write_charge_file(out_file, residues, charges, groups, model, source)
    return charges, groups


def cli_parser(prog="protein_charges.py"):
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Generate the protein charge file used by the DNANM/DNACT screened electrostatics "
                    "(protein_charge_file). Charges are either assigned from the sequence or read from a "
                    "PQR file produced by PDB2PQR/PROPKA.")
    parser.add_argument('topology', type=str, help="The DNANM/DNACT topology file")
    parser.add_argument('out_file', type=str, help="The charge file to write")
    parser.add_argument('--pqr', type=str, default=None,
                        help="PQR file with the same residues as the topology, in the same order. "
                             "If given, its per-residue charges replace the sequence-based model")
    parser.add_argument('--pqr-chain-order', type=str, default=None,
                        help="Comma-separated PQR chain IDs in peptide-strand order, e.g. A,B. Required for ambiguous repeated sequences.")
    parser.add_argument('-q', '--quiet', action='store_true',
                        help="Suppress informational messages")
    parser.add_argument('--his-charge', type=float, default=0.0,
                        help="Charge of histidine in the sequence-based model (default: 0.0)")
    parser.add_argument('--termini', action='store_true',
                        help="Add +1/-1 on the first/last bead of every strand in the sequence-based model")
    parser.add_argument('-g', '--group-by', type=str, choices=['strand', 'single', 'residue'], default='strand',
                        help="How to define charge groups: one per topology strand (default), a single group, "
                             "or one per bead. Beads in the same group do not interact electrostatically "
                             "when exclude_intragroup_electrostatics is true")
    parser.add_argument('--group-file', type=str, default=None,
                        help="Text file with one integer group id per protein bead; overrides --group-by")
    return parser


def main():
    parser = cli_parser(os.path.basename(__file__))
    args = parser.parse_args()
    logger_settings.set_quiet(args.quiet)
    try:
        charges, groups = protein_charges(args.topology, args.out_file, args.pqr, args.his_charge,
                                          args.termini, args.group_by, args.group_file,
                                          args.pqr_chain_order)
    except (ChargeError, OSError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    log(f"Wrote {len(charges)} protein charges in {len(set(groups))} group(s) to {args.out_file}")
    log(f"Net protein charge: {sum(charges):+.3f} e")
    log(f"Set protein_electrostatics = true and protein_charge_file = {args.out_file} in oxDNA; CPU only.")


if __name__ == '__main__':
    main()
