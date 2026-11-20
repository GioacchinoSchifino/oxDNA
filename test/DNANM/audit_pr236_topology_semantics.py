#!/usr/bin/env python3
"""Regression: ANM new-style topology must decode numeric base tokens.

Unmodified Erik PR #236 physics. Report failures as diagnostics, not as
breakage of the unrelated standard DNA2 model. Golden control DNA2 must work.
"""
from pathlib import Path
import subprocess, tempfile
REPO=Path(__file__).resolve().parents[2]

def run(model, topology):
    with tempfile.TemporaryDirectory(prefix="pr236-btype-") as tmp:
        d=Path(tmp)
        (d/"system.top").write_text(topology)
        n=topology.count("type=")
        (d/"init.dat").write_text("t = 0\nb = 30 30 30\nE = 0 0 0\n"+
            "\n".join(f"{5+i*5} 5 5 1 0 0 0 0 1 0.001 0.002 0.003 0.001 0.002 0.003"
                      for i in range(n))+"\n")
        (d/"springs.par").write_text("1\n")
        (d/"input").write_text(f"""backend = CPU
interaction_type = {model}
sim_type = MD
steps = 1
T = 300K
salt_concentration = 1.0
dt = 0.0001
thermostat = no
refresh_vel = 1
seed = 11
restart_step_counter = 1
time_scale = linear
verlet_skin = 0.25
parfile = springs.par
topology = system.top
conf_file = init.dat
trajectory_file = trajectory.dat
lastconf_file = lastconf.dat
log_file = log.dat
energy_file = energy.dat
no_stdout_energy = 1
print_conf_interval = 1
print_energy_every = 1
""")
        p=subprocess.run([str(REPO/"build"/"bin"/"oxDNA"),"input"],
                         cwd=d, capture_output=True, text=True, timeout=45)
        return p.returncode,(d/"log.dat").read_text(errors="replace") if (d/"log.dat").exists() else ""

def main():
    dna_top="2 2 5->3\nA type=DNA circular=false\n(0) type=DNA circular=false\n"
    mixed_top="3 3 2 1 2 5->3\nA type=DNA circular=false\n(0) type=DNA circular=false\nK type=peptide circular=false\n"
    rc,log=run("DNA2",dna_top)
    if rc: raise AssertionError("Standard DNA2 control rejects numeric base, invalid diagnostic: "+log[-900:])
    print("CONTROL PASS: DNA2 accepts new-style (0) base token",flush=True)
    rc,log=run("DNANM",mixed_top)
    if rc:
        raise AssertionError("DNANM must accept numeric-base tokens like DNA2: "+log[-1000:])
    print("PASS: DNANM and DNA2 both accept a numeric DNA base token",flush=True)

if __name__=="__main__":main()
