import argparse
import math
from oxDNA_analysis_tools.UTILS.logger import log, logger_settings
import os
import sys
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
    """Return (canonical protein strand ID, residue) for each ANM bead.

    Uses RyeReader.strand_describe, as requested in the upstream OAT review,
    and accepts both legacy and type=peptide 5->3 topologies. For new-style
    topologies, the charge-file strand identifiers are -1, -2, ... in
    peptide-strand order, matching the CPU C++ charge reader.
    """
    from oxDNA_analysis_tools.UTILS.RyeReader import strand_describe
    try:
        with open(topology) as f:
            header = f.readline().split()
        if not header:
            raise ChargeError(f"Topology '{topology}' is empty")
        new_style = header[-1] == "5->3"
        if new_style:
            if len(header) not in (3, 6):
                raise ChargeError("Invalid new-style ANM header (expected N, N_strands, [N_DNA, N_protein, N_DNA_strands], 5->3)")
            expected_proteins = int(header[3]) if len(header) == 6 else None
        else:
            if len(header) != 5:
                raise ChargeError("Old DNANM/DNACT topology requires N N_strands N_DNA N_protein N_DNA_strands")
            expected_proteins = int(header[3])
        expected_n = int(header[0])
        system, monomers = strand_describe(topology)
        if len(monomers) != expected_n:
            raise ChargeError(f"Topology header has {expected_n} particles but RyeReader found {len(monomers)}")
        proteins = []
        peptide_number = 0
        seen_types = []
        for strand in system.strands:
            seen_types.append(strand.is_peptide())
            if not strand.is_peptide():
                continue
            peptide_number += 1
            label = -peptide_number if new_style else int(strand.id)
            for m in strand.monomers:
                aa = str(m.btype).upper()
                if len(aa) != 1 or not aa.isalpha():
                    raise ChargeError(f"Invalid protein residue '{aa}' in strand {label}")
                proteins.append((label, aa))
        if sum(x != y for x, y in zip(seen_types[1:], seen_types[:-1])) > 1:
            raise ChargeError("ANM topology must put amino acids in one contiguous block")
        if expected_proteins is not None and expected_proteins != len(proteins):
            raise ChargeError(f"Topology declares {expected_proteins} protein beads but has {len(proteins)}")
        if not proteins:
            raise ChargeError("The topology contains no protein beads")
        return proteins
    except ChargeError:
        raise
    except (OSError, ValueError, IndexError, RuntimeError) as e:
        raise ChargeError(f"Cannot read ANM topology '{topology}': {e}") from e

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


def read_pqr_residue_charges(pqr: str, with_chains: bool = False) -> List[Tuple[str, float]]:
    """
        Sum the atomic charges of each residue in a PQR file (as written by PDB2PQR with PROPKA).
        Because the charges come from the protonated structure, they already include the
        predicted protonation states.

        Parameters:
            pqr (str): Path to the PQR file.

        Returns:
            List[Tuple[str, float]]: (one-letter residue code, net charge) per residue, in file order.
    """
    residues: "OrderedDict[tuple, List]" = OrderedDict()
    with open(pqr) as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue
            t = line.split()
            # ATOM serial name resname [chain] resseq x y z charge radius
            if len(t) < 10:
                raise ChargeError(f"Cannot parse PQR line: {line.rstrip()}")
            resname = t[3].upper()
            try:
                float(t[4])
                chain, resseq = '', t[4]
            except ValueError:
                chain, resseq = t[4], t[5]
            try:
                charge = float(t[-2])
            except ValueError:
                raise ChargeError(f"Cannot read the charge on PQR line: {line.rstrip()}")
            if resname not in THREE_TO_ONE:
                raise ChargeError(f"Unsupported residue '{resname}' in PQR file")
            key = (chain, resseq)
            if key not in residues:
                residues[key] = [THREE_TO_ONE[resname], 0.0]
            residues[key][1] += charge
    if not residues:
        raise ChargeError(f"No ATOM records found in '{pqr}'")
    if with_chains:
        return [(key[0], value[0], value[1]) for key, value in residues.items()]
    return [(code, q) for code, q in residues.values()]


def pqr_charges(residues: Sequence[Tuple[int, str]], pqr: str, chain_order: Optional[Sequence[str]] = None) -> List[float]:
    """
        Map per-residue PQR charges onto the protein beads. The PQR must contain exactly the
        residues of the topology, in the same order (this is checked residue by residue).

        Parameters:
            residues (Sequence[Tuple[int, str]]): (strand id, residue code) for each protein bead.
            pqr (str): Path to the PQR file.

        Returns:
            List[float]: One charge (in units of e) per protein bead.
    """
    if chain_order is not None:
        top_chains = list(OrderedDict.fromkeys(s for s, _ in residues))
        if len(top_chains) != len(chain_order) or len(set(chain_order)) != len(chain_order):
            raise ChargeError("--pqr-chain-order must provide one unique PQR chain ID per peptide strand")
        rows = read_pqr_residue_charges(pqr, with_chains=True)
        pqr_chains = OrderedDict()
        for chain, aa, q in rows:
            pqr_chains.setdefault(chain, []).append((aa, q))
        if set(pqr_chains) != set(chain_order):
            raise ChargeError(f"PQR chains {list(pqr_chains)} do not match --pqr-chain-order {list(chain_order)}")
        top_by_chain = OrderedDict((s, []) for s in top_chains)
        for strand, aa in residues:
            top_by_chain[strand].append(aa)
        out = []
        for strand, pqr_id in zip(top_chains, chain_order):
            expected = top_by_chain[strand]
            found = pqr_chains[pqr_id]
            if len(found) != len(expected) or any(a != b for a, (b, _) in zip(expected, found)):
                raise ChargeError(f"PQR chain {pqr_id} does not match peptide strand {strand} by sequence and length")
            out.extend(q for _, q in found)
        return out

    # Repeated identical subunit sequences cannot be matched uniquely by
    # sequence/order alone when PQR has multiple chains and distinct charges.
    top_by_chain = OrderedDict()
    for strand, aa in residues:
        top_by_chain.setdefault(strand, []).append(aa)
    duplicate_sequences = len({tuple(seq) for seq in top_by_chain.values()}) < len(top_by_chain)
    pqr_rows = read_pqr_residue_charges(pqr, with_chains=True)
    if duplicate_sequences and len({c for c, _, _ in pqr_rows}) > 1:
        raise ChargeError("Ambiguous identical protein subunits in PQR: use --pqr-chain-order to map each PQR chain explicitly")

    pqr_res = read_pqr_residue_charges(pqr)
    if len(pqr_res) != len(residues):
        raise ChargeError(
            f"The PQR file has {len(pqr_res)} residues but the topology has {len(residues)} protein beads. "
            "They must describe the same residues in the same order")
    for i, ((_, top_code), (pqr_code, _)) in enumerate(zip(residues, pqr_res)):
        if top_code != pqr_code:
            raise ChargeError(
                f"Residue mismatch at protein index {i}: topology has '{top_code}', PQR has '{pqr_code}'")
    return [q for _, q in pqr_res]


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
    if any(not math.isfinite(q) for q in charges):
        raise ChargeError("Protein charges must be finite")
    if any(not isinstance(g, int) or g < 0 for g in groups):
        raise ChargeError("Protein charge group IDs must be non-negative integers")
    # Metadata is parsed as 'key = value' per line, so keep values on one line.
    model = ' '.join(model.split())
    source = ' '.join(source.split())
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
                    group_file: Optional[str] = None, pqr_chain_order: Optional[Sequence[str]] = None) -> Tuple[List[float], List[int]]:
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
    if not math.isfinite(his_charge):
        raise ChargeError("Histidine charge must be finite")
    if pqr_chain_order is not None and pqr is None:
        raise ChargeError("--pqr-chain-order requires --pqr")

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
                        help="Comma-separated PQR chain IDs in peptide-strand order; required for ambiguous homomers")
    parser.add_argument('-q', '--quiet', action='store_true',
                        help="Suppress informational messages (standard OAT logger)")
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
                                          args.pqr_chain_order.split(',') if args.pqr_chain_order else None)
    except (ChargeError, OSError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    log(f"Wrote {len(charges)} protein charges in {len(set(groups))} group(s) to {args.out_file}")
    log(f"Net protein charge: {sum(charges):+.3f} e")
    log("Set protein_electrostatics = true and "
          f"protein_charge_file = {args.out_file} in the oxDNA input file.")


if __name__ == '__main__':
    main()
