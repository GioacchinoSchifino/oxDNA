#!/usr/bin/env python3
"""Nonfinite / malformed ANMT parameter parsing audit of original PR #236."""
import math
import subprocess
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/"examples"/"DNANM"/"KDPG"
SOURCE=(DATA/"kdpg_act.par").read_text().splitlines()
assert len(SOURCE[1].split())>=9

def run(label,token):
    with tempfile.TemporaryDirectory(prefix="pr236-anmt-parser-") as td:
        d=Path(td)
        lines=SOURCE[:]
        if token is not None:
            p=lines[1].split()
            p[5]=token # first cosine in first spring of real KDPG parameter file
            lines[1]=" ".join(p)
        (d/"test.par").write_text("\n".join(lines)+"\n")
        (d/"input").write_text(f"""backend = CPU
interaction_type = DNACT
salt_concentration = 1.0
sim_type = MD
steps = 0
thermostat = no
T = 300K
dt = 0.001
verlet_skin = 0.05
refresh_vel = 0
seed = 1
restart_step_counter = 1
time_scale = linear
bending_k = 50.0
torsion_k = 50.0
parfile = test.par
topology = {DATA/'generated.top'}
conf_file = {DATA/'s1.dat'}
trajectory_file = trajectory.dat
lastconf_file = lastconf.dat
log_file = log.dat
energy_file = energy.dat
no_stdout_energy = 1
print_conf_interval = 1
print_energy_every = 1
""")
        p=subprocess.run([str(ROOT/"build"/"bin"/"oxDNA"),"input"],cwd=d,capture_output=True,text=True,timeout=60)
        log=(d/"log.dat").read_text(errors="replace") if (d/"log.dat").exists() else ""
        energies=(d/"energy.dat").read_text(errors="replace") if (d/"energy.dat").exists() else ""
        print(f"{label}: exit={p.returncode}; tail-log={log[-460:].replace(chr(10),' | ')}; energies={energies[:200].replace(chr(10),' | ')}",flush=True)
        return p.returncode,log

def main():
    ctrl=run("VALID_CONTROL",None)
    if ctrl[0]!=0:
        raise RuntimeError("Control DNACT failed; no inference about parser input safety")
    print("CONTROL PASS",flush=True)
    for label,value in [("NONFINITE_COSINE","nan"),("COSINE_TRAILING_GARBAGE","0.5junk")]:
        rc,log=run(label,value)
        if rc==0:
            print(f"CONFIRMED_INVALID_ACCEPTED {label}: malformed ANMT parameter was not rejected",flush=True)
        else:
            print(f"INVALID_REJECTED {label}: {log[-350:].replace(chr(10),' | ')}",flush=True)

if __name__=="__main__":main()
