#!/usr/bin/env python3
"""Independent *diagnostic* of pressure/stress accounting in hybrid ANM.

Two protein beads joined by one stretched spring, no nucleotides and no
nonbonded contact. The inherited DNAInteraction custom-stress path should
include the same spring virial as the explicit standard-stress recomputation.
This script reports any difference; --require-match turns it into a regression.
"""
import argparse
import math
import subprocess
import tempfile
from pathlib import Path

def pressure(binary, custom):
    with tempfile.TemporaryDirectory(prefix="pr236-anm-pressure-") as tmp:
        p = Path(tmp)
        (p / "system.top").write_text("2 1 0 2 0 5->3\nAK type=peptide circular=false\n")
        (p / "springs.par").write_text("2\n0 1 0.6 s 10.0\n")
        (p / "init.dat").write_text(
            "t = 0\nb = 10 10 10\nE = 0 0 0\n"
            "3 5 5 1 0 0 0 0 1 0 0 0 0 0 0\n"
            "4 5 5 1 0 0 0 0 1 0 0 0 0 0 0\n"
        )
        (p / "input").write_text(f"""backend = CPU
interaction_type = DNANM
sim_type = MD
steps = 1
T = 300K
dt = 0.0000001
thermostat = no
refresh_vel = 0
seed = 17
restart_step_counter = 1
time_scale = linear
verlet_skin = 0.2
salt_concentration = 1.0
parfile = springs.par
topology = system.top
conf_file = init.dat
trajectory_file = traj.dat
lastconf_file = lastconf.dat
log_file = run.log
energy_file = energy.dat
no_stdout_energy = 1
print_energy_every = 1
print_conf_interval = 1
stress_tensor_no_kinetic_part = true
data_output_1 = {{
    name = pressure.dat
    print_every = 1
    col_1 = {{
        type = pressure
        custom_stress_tensor = {"true" if custom else "false"}
    }}
}}
""")
        result = subprocess.run([str(binary), "input"], cwd=p, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=90)
        log=(p/"run.log").read_text(errors="replace") if (p/"run.log").exists() else ""
        if result.returncode:
            raise RuntimeError("oxDNA failed: " + log[-1800:] + result.stderr[-500:])
        outputs=[]
        for line in (p/"pressure.dat").read_text().splitlines():
            try:
                val=float(line.split()[0])
                if math.isfinite(val):
                    outputs.append(val)
            except (ValueError, IndexError):
                pass
        if not outputs:
            raise RuntimeError("No finite pressure values in pressure.dat: " + log[-700:])
        return outputs[0]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--binary", required=True, type=Path)
    ap.add_argument("--require-match", action="store_true",
                    help="assert the two supported stress paths agree")
    args=ap.parse_args()
    binary=args.binary.resolve()
    custom=pressure(binary, True)
    standard=pressure(binary, False)
    err=abs(custom-standard)
    print(f"ANM_PRESSURE custom={custom:.12g} standard={standard:.12g} absolute_difference={err:.12g}", flush=True)
    if err > 1e-6:
        print("POTENTIAL_MISSING_ANM_CUSTOM_VIRIAL: custom stress disagrees with independently recomputed standard virial", flush=True)
        if args.require_match:
            raise AssertionError("ANM stress pathways disagree")
    else:
        print("PASS_PRESSURE_PATH_CONSISTENCY for this fixture", flush=True)

if __name__=="__main__":
    main()
