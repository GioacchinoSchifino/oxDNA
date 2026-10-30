#!/usr/bin/env python3

"""
Prepare and validate DNANM/DNACT protein electrostatic charge files.

This utility intentionally does not assign protonation states, infer pKa
values, guess residue charges, or otherwise define a chemical charge model.

The prepare command combines:

    1. an oxDNA mixed DNANM/DNACT topology, and
    2. an explicit user-supplied assignment file

whose data rows are:

    protein_local_index charge_e group_id

into the runtime format:

    # DNANM_ES_CHARGES 1

The validate command checks the resulting file against the topology using
the same indexing and identity assumptions as DNANMInteraction.

Only the Python standard library is required.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


FORMAT_HEADER = "# DNANM_ES_CHARGES 1"


class ChargeFileError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProteinRow:
    local_index: int
    global_index: int
    topology_strand: int
    residue: str


@dataclass(frozen=True)
class Assignment:
    local_index: int
    charge_e: float
    group_id: int


@dataclass(frozen=True)
class ChargeRecord:
    local_index: int
    topology_strand: int
    residue: str
    charge_e: float
    group_id: int


@dataclass(frozen=True)
class ParsedTopology:
    n_particles: int
    n_strands: int
    ndna: int
    npro: int
    ndnas: int
    proteins: Tuple[ProteinRow, ...]


@dataclass(frozen=True)
class ParsedChargeFile:
    metadata: Dict[str, str]
    records: Tuple[ChargeRecord, ...]


def fail(message: str) -> None:
    raise ChargeFileError(message)


def parse_int(text: str, what: str) -> int:
    try:
        return int(text, 10)
    except ValueError:
        fail(f"{what} must be an integer, got '{text}'")


def parse_finite_float(text: str, what: str) -> float:
    try:
        value = float(text)
    except ValueError:
        fail(f"{what} must be a floating-point number, got '{text}'")

    if not math.isfinite(value):
        fail(
            f"{what} must be finite; "
            f"NaN and infinity are not permitted"
        )

    return value


def read_topology(path: Path) -> ParsedTopology:
    try:
        raw_lines = path.read_text().splitlines()
    except OSError as exc:
        fail(f"could not read topology '{path}': {exc}")

    if not raw_lines:
        fail(f"topology '{path}' is empty")

    header = raw_lines[0].split()

    if len(header) != 5:
        fail(
            "DNANM/DNACT mixed topology header must contain exactly "
            "five integers: N N_strands ndna npro ndnas"
        )

    values = tuple(
        parse_int(token, f"topology header field {i + 1}")
        for i, token in enumerate(header)
    )

    (
        n_particles,
        n_strands,
        ndna,
        npro,
        ndnas,
    ) = values

    if n_particles < 0:
        fail("topology N must be non-negative")

    if n_strands < 0 or n_strands > n_particles:
        fail("invalid N_strands in topology header")

    if ndna < 0 or ndna > n_particles:
        fail("invalid ndna in topology header")

    if npro < 0 or npro > n_particles:
        fail("invalid npro in topology header")

    if ndnas < 0 or ndnas > n_particles:
        fail("invalid ndnas in topology header")

    particle_rows: List[List[str]] = []

    for line_number, line in enumerate(
        raw_lines[1:],
        start=2,
    ):
        stripped = line.strip()

        if not stripped or stripped.startswith("#"):
            continue

        fields = stripped.split()

        if not fields:
            continue

        particle_rows.append(fields)

    if len(particle_rows) != n_particles:
        fail(
            f"topology declares N={n_particles} but contains "
            f"{len(particle_rows)} particle rows"
        )

    proteins: List[ProteinRow] = []

    for global_index, fields in enumerate(particle_rows):
        strand = parse_int(
            fields[0],
            f"topology particle {global_index} strand",
        )

        if strand == 0:
            fail(
                f"topology particle {global_index} uses forbidden strand 0"
            )

        if strand < 0:
            if len(fields) < 4:
                fail(
                    f"protein topology row {global_index} is malformed; "
                    "expected at least strand residue nside cside"
                )

            residue = fields[1]

            if len(residue) != 1:
                fail(
                    "protein electrostatics requires one-letter "
                    f"amino-acid codes; particle {global_index} has "
                    f"'{residue}'"
                )

            residue = residue.upper()

            proteins.append(
                ProteinRow(
                    local_index=len(proteins),
                    global_index=global_index,
                    topology_strand=strand,
                    residue=residue,
                )
            )

        else:
            if len(fields) < 4:
                fail(
                    f"DNA topology row {global_index} is malformed; "
                    "expected at least strand base n3 n5"
                )

    if len(proteins) != npro:
        fail(
            f"topology contains {len(proteins)} protein rows "
            f"but header declares npro={npro}"
        )

    return ParsedTopology(
        n_particles=n_particles,
        n_strands=n_strands,
        ndna=ndna,
        npro=npro,
        ndnas=ndnas,
        proteins=tuple(proteins),
    )


def read_assignments(
    path: Path,
    topology: ParsedTopology,
) -> Tuple[Assignment, ...]:
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        fail(f"could not read assignments '{path}': {exc}")

    assignments: Dict[int, Assignment] = {}

    for line_number, line in enumerate(
        lines,
        start=1,
    ):
        stripped = line.strip()

        if not stripped or stripped.startswith("#"):
            continue

        fields = stripped.split()

        if len(fields) != 3:
            fail(
                f"assignment line {line_number} must contain exactly "
                "three fields: protein_local_index charge_e group_id"
            )

        local_index = parse_int(
            fields[0],
            f"assignment line {line_number} protein_local_index",
        )

        charge_e = parse_finite_float(
            fields[1],
            f"assignment line {line_number} charge_e",
        )

        group_id = parse_int(
            fields[2],
            f"assignment line {line_number} group_id",
        )

        if local_index < 0 or local_index >= topology.npro:
            fail(
                f"assignment line {line_number} protein_local_index "
                f"{local_index} is outside 0..{topology.npro - 1}"
            )

        if group_id < 0:
            fail(
                f"assignment line {line_number} group_id "
                "must be non-negative"
            )

        if local_index in assignments:
            fail(
                f"duplicate assignment for protein_local_index "
                f"{local_index}"
            )

        assignments[local_index] = Assignment(
            local_index=local_index,
            charge_e=charge_e,
            group_id=group_id,
        )

    missing = sorted(
        set(range(topology.npro))
        -
        set(assignments)
    )

    if missing:
        fail(
            "assignments are missing protein_local_index values: "
            +
            ", ".join(str(x) for x in missing)
        )

    return tuple(
        assignments[i]
        for i in range(topology.npro)
    )


def validate_metadata(
    metadata: Dict[str, str],
    topology: ParsedTopology,
) -> None:
    required = (
        "index_space",
        "index_base",
        "n_protein_beads",
    )

    for key in required:
        if key not in metadata:
            fail(
                f"version-1 charge file is missing metadata '{key}'"
            )

    if metadata["index_space"] != "protein_local":
        fail(
            "version-1 charge file must use "
            "index_space = protein_local"
        )

    if metadata["index_base"] != "0":
        fail(
            "version-1 charge file must use index_base = 0"
        )

    declared_npro = parse_int(
        metadata["n_protein_beads"],
        "n_protein_beads",
    )

    if declared_npro != topology.npro:
        fail(
            f"charge file declares n_protein_beads={declared_npro} "
            f"but topology contains {topology.npro}"
        )


def read_charge_file(
    path: Path,
    topology: ParsedTopology,
) -> ParsedChargeFile:
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        fail(f"could not read charge file '{path}': {exc}")

    meaningful = [
        line.strip()
        for line in lines
        if line.strip()
    ]

    if not meaningful:
        fail(f"charge file '{path}' is empty")

    if meaningful[0] != FORMAT_HEADER:
        fail(
            "charge file must begin with exactly "
            "'# DNANM_ES_CHARGES 1'"
        )

    metadata: Dict[str, str] = {}
    records: List[ChargeRecord] = []

    header_seen = False

    for line_number, line in enumerate(
        lines,
        start=1,
    ):
        stripped = line.strip()

        if not stripped:
            continue

        if stripped == FORMAT_HEADER:
            if header_seen:
                fail(
                    f"duplicate format header on line {line_number}"
                )

            header_seen = True
            continue

        if stripped.startswith("#"):
            body = stripped[1:].strip()

            if not body:
                continue

            if "=" not in body:
                # Free-form comments are allowed.
                continue

            key, value = body.split("=", 1)

            key = key.strip()
            value = value.strip()

            if not key:
                fail(
                    f"empty metadata key on line {line_number}"
                )

            if key in metadata:
                fail(
                    f"duplicate metadata key '{key}' "
                    f"on line {line_number}"
                )

            metadata[key] = value
            continue

        fields = stripped.split()

        if len(fields) != 5:
            fail(
                f"charge record line {line_number} must contain exactly "
                "five fields: local_index topology_strand residue "
                "charge_e group_id"
            )

        local_index = parse_int(
            fields[0],
            f"charge record line {line_number} local_index",
        )

        topology_strand = parse_int(
            fields[1],
            f"charge record line {line_number} topology_strand",
        )

        residue = fields[2]

        if len(residue) != 1:
            fail(
                f"charge record line {line_number} residue "
                "must be a one-letter amino-acid code"
            )

        residue = residue.upper()

        charge_e = parse_finite_float(
            fields[3],
            f"charge record line {line_number} charge_e",
        )

        group_id = parse_int(
            fields[4],
            f"charge record line {line_number} group_id",
        )

        if group_id < 0:
            fail(
                f"charge record line {line_number} group_id "
                "must be non-negative"
            )

        records.append(
            ChargeRecord(
                local_index=local_index,
                topology_strand=topology_strand,
                residue=residue,
                charge_e=charge_e,
                group_id=group_id,
            )
        )

    if not header_seen:
        fail(
            "charge file is missing '# DNANM_ES_CHARGES 1'"
        )

    validate_metadata(
        metadata,
        topology,
    )

    by_local: Dict[int, ChargeRecord] = {}

    for record in records:
        i = record.local_index

        if i < 0 or i >= topology.npro:
            fail(
                f"charge record local_index {i} is outside "
                f"0..{topology.npro - 1}"
            )

        if i in by_local:
            fail(
                f"duplicate charge record for local_index {i}"
            )

        topo = topology.proteins[i]

        if record.topology_strand != topo.topology_strand:
            fail(
                f"local_index {i} topology_strand mismatch: "
                f"charge file has {record.topology_strand}, "
                f"topology has {topo.topology_strand}"
            )

        if record.residue != topo.residue:
            fail(
                f"local_index {i} residue mismatch: "
                f"charge file has {record.residue}, "
                f"topology has {topo.residue}"
            )

        by_local[i] = record

    missing = sorted(
        set(range(topology.npro))
        -
        set(by_local)
    )

    if missing:
        fail(
            "charge file is missing local_index values: "
            +
            ", ".join(str(x) for x in missing)
        )

    if len(records) != topology.npro:
        fail(
            f"charge file contains {len(records)} records "
            f"for topology npro={topology.npro}"
        )

    if "n_groups" in metadata:
        declared_groups = parse_int(
            metadata["n_groups"],
            "n_groups",
        )

        if declared_groups < 0:
            fail("n_groups must be non-negative")

        observed_groups = len(
            {
                record.group_id
                for record in records
            }
        )

        if declared_groups != observed_groups:
            fail(
                f"charge file declares n_groups={declared_groups} "
                f"but contains {observed_groups} distinct groups"
            )

    return ParsedChargeFile(
        metadata=metadata,
        records=tuple(
            by_local[i]
            for i in range(topology.npro)
        ),
    )


def write_charge_file(
    output: Path,
    topology: ParsedTopology,
    assignments: Tuple[Assignment, ...],
    charge_model: str,
    charge_source: str,
) -> None:
    if not charge_model.strip():
        fail("charge_model must not be empty")

    if not charge_source.strip():
        fail("charge_source must not be empty")

    if "\n" in charge_model or "\r" in charge_model:
        fail("charge_model must not contain newlines")

    if "\n" in charge_source or "\r" in charge_source:
        fail("charge_source must not contain newlines")

    group_count = len(
        {
            assignment.group_id
            for assignment in assignments
        }
    )

    lines = [
        FORMAT_HEADER,
        "# index_space = protein_local",
        "# index_base = 0",
        f"# charge_model = {charge_model}",
        f"# charge_source = {charge_source}",
        f"# n_protein_beads = {topology.npro}",
        f"# n_groups = {group_count}",
        (
            "# columns = protein_index topology_strand "
            "residue charge_e group_id"
        ),
    ]

    for topo, assignment in zip(
        topology.proteins,
        assignments,
    ):
        lines.append(
            f"{topo.local_index} "
            f"{topo.topology_strand} "
            f"{topo.residue} "
            f"{assignment.charge_e:.17g} "
            f"{assignment.group_id}"
        )

    text = "\n".join(lines) + "\n"

    try:
        output.write_text(text)
    except OSError as exc:
        fail(
            f"could not write output '{output}': {exc}"
        )


def command_prepare(args: argparse.Namespace) -> int:
    topology = read_topology(
        Path(args.topology)
    )

    assignments = read_assignments(
        Path(args.assignments),
        topology,
    )

    output = Path(args.output)

    write_charge_file(
        output,
        topology,
        assignments,
        args.charge_model,
        args.charge_source,
    )

    # Always validate our own output through the independent reader.
    parsed = read_charge_file(
        output,
        topology,
    )

    net_charge = sum(
        record.charge_e
        for record in parsed.records
    )

    groups = len(
        {
            record.group_id
            for record in parsed.records
        }
    )

    print(
        "PREPARED"
        f" output={output}"
        f" n_protein_beads={len(parsed.records)}"
        f" n_groups={groups}"
        f" net_charge_e={net_charge:.17g}"
    )

    return 0


def command_validate(args: argparse.Namespace) -> int:
    topology = read_topology(
        Path(args.topology)
    )

    parsed = read_charge_file(
        Path(args.charges),
        topology,
    )

    net_charge = sum(
        record.charge_e
        for record in parsed.records
    )

    max_abs_charge = max(
        (
            abs(record.charge_e)
            for record in parsed.records
        ),
        default=0.0,
    )

    groups = len(
        {
            record.group_id
            for record in parsed.records
        }
    )

    print(
        "VALID"
        f" charges={args.charges}"
        f" n_protein_beads={len(parsed.records)}"
        f" n_groups={groups}"
        f" net_charge_e={net_charge:.17g}"
        f" max_abs_charge_e={max_abs_charge:.17g}"
    )

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare or validate DNANM/DNACT protein "
            "electrostatic charge files."
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    prepare = subparsers.add_parser(
        "prepare",
        help=(
            "Create a DNANM_ES_CHARGES v1 file from topology "
            "plus explicit local-index charge/group assignments."
        ),
    )

    prepare.add_argument(
        "--topology",
        required=True,
        help="Mixed DNANM/DNACT topology file.",
    )

    prepare.add_argument(
        "--assignments",
        required=True,
        help=(
            "Text file with rows: "
            "protein_local_index charge_e group_id."
        ),
    )

    prepare.add_argument(
        "--output",
        required=True,
        help="Output DNANM_ES_CHARGES v1 file.",
    )

    prepare.add_argument(
        "--charge-model",
        required=True,
        help=(
            "Human-readable charge-model provenance. "
            "No model is inferred by this utility."
        ),
    )

    prepare.add_argument(
        "--charge-source",
        required=True,
        help="Human-readable source/provenance string.",
    )

    prepare.set_defaults(
        handler=command_prepare
    )

    validate = subparsers.add_parser(
        "validate",
        help=(
            "Fail-closed validation of a DNANM_ES_CHARGES "
            "v1 file against a topology."
        ),
    )

    validate.add_argument(
        "--topology",
        required=True,
        help="Mixed DNANM/DNACT topology file.",
    )

    validate.add_argument(
        "--charges",
        required=True,
        help="DNANM_ES_CHARGES v1 file.",
    )

    validate.set_defaults(
        handler=command_validate
    )

    return parser


def main() -> int:
    parser = build_parser()

    args = parser.parse_args()

    try:
        return args.handler(args)
    except ChargeFileError as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
