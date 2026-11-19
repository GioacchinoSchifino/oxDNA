#!/usr/bin/env python3
"""Minimal CPU-only smoke and parser regression checks for the #236 ES port.
Runs against the KDPG example (legacy/new topology) with pinned coordinates.
This does NOT replace force finite-difference or long-time dynamics tests.
"""
import argparse
import math
import os
from pathlib import Path
import subprocess
import tempfile

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "examples" / "DNANM" / "KDPG"
NEW = REPO / "test" / "DNANM" / "new_topology" / "kdpg_new.top"

def rows_for_topology(path):
    raw = [x.strip() for x in path.read_text().splitlines() if x.strip() and not x.startswith("#")]
    if raw[0].endswith("5->3"):
        residues = []
        ordinal = 0
        for row in raw[1:]:
            fields = row.split()
            if "type=peptide" in fields[1:]:
                ordinal += 1
                residues.extend([(-ordinal, x) for x in fields[0]])
    else:
        residues = [(int(fields[0]), fields[1])
                    for row in raw[1:]
                    if (fields := row.split()) and int(fields[0]) < 0]
    return residues

def write_charges(topology, path, alternating):
    residues = rows_for_topology(topology)
    lines = ["# DNANM_ES_CHARGES 1",
             "# index_space = protein_local", "# index_base = 0",
             f"# n_protein_beads = {len(residues)}",
             f"# n_groups = {len(set(s for s, _ in residues))}"]
    groups = {s: i for i, s in enumerate(dict.fromkeys(s for s, _ in residues))}
    for i, (s, aa) in enumerate(residues):
        q = (1.0 if i % 2 else -1.0) if alternating else 0.0
        lines.append(f"{i} {s} {aa} {q:.1f} {groups[s]}")
    path.write_text("\n".join(lines) + "\n")

def run_one(binary, top, interaction, enabled, alternating):
    with tempfile.TemporaryDirectory(prefix="anm-es-") as td:
        wd = Path(td)
        charges = wd / "charges.dat"
        write_charges(top, charges, alternating)
        par = DATA / ("kdpg_act.par" if interaction == "DNACT" else "kdpg_anm.par")
        text = f"""backend = CPU
backend_precision = double
interaction_type = {interaction}
salt_concentration = 1.0
sim_type = MD
steps = 0
thermostat = john
T = 300K
dt = 0.001
verlet_skin = 0.05
diff_coeff = 2.5
newtonian_steps = 103
parfile = {par}
topology = {top}
conf_file = {DATA / 's1.dat'}
trajectory_file = {wd / 'trajectory.dat'}
lastconf_file = {wd / 'lastconf.dat'}
log_file = {wd / 'log.dat'}
energy_file = {wd / 'energy.dat'}
print_conf_interval = 1000
print_energy_every = 1
time_scale = linear
refresh_vel = 1
seed = 1
restart_step_counter = 1
bending_k = 50.0
torsion_k = 50.0
protein_electrostatics = {'true' if enabled else 'false'}
protein_charge_file = {charges}
exclude_intragroup_electrostatics = false
"""
        (wd / "input").write_text(text)
        p = subprocess.run([str(binary), "input"], cwd=wd, capture_output=True, text=True, timeout=120)
        if p.returncode:
            raise RuntimeError(f"oxDNA failed: {interaction} {top.name} ES={enabled}: exit={p.returncode} LOG={(wd / 'log.dat').read_text()[-3200:] if (wd / 'log.dat').exists() else '<no log>'} STDERR={p.stderr[-2200:]} STDOUT={p.stdout[-1000:]}")
        # A run must create a finite energy trace.
        path = wd / "energy.dat"
        if not path.exists() or not path.stat().st_size:
            raise RuntimeError(f"Missing energy output for {interaction} {top.name}")
        lines = [line for line in path.read_text().splitlines() if line and not line.startswith("#")]
        if not lines: raise RuntimeError("Empty energy trace")
        vals = [float(v) for v in lines[0].split()]
        if not all(math.isfinite(v) for v in vals): raise RuntimeError("Nonfinite energies")
        return vals

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve()
    for top in [DATA / "generated.top", NEW]:
        for interaction in ["DNANM", "DNACT"]:
            baseline = run_one(binary, top, interaction, False, False)
            zero = run_one(binary, top, interaction, True, False)
            charged = run_one(binary, top, interaction, True, True)
            assert baseline == zero, f"Zero-charge regression on {top}, {interaction}: {baseline} vs {zero}"
            print(f"PASS {top.name} {interaction}: ES-OFF == ES-ON-zero; charged finite {charged}")
    print("CPU SMOKE PASS (not full physics validation)")

if __name__ == "__main__": main()
