#!/usr/bin/env python3
"""Regression: invalid nonfinite or partially parsed ANMT parameters must fail."""
import math
import subprocess
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/"examples"/"DNANM"/"KDPG"
SOURCE=(DATA/"kdpg_act.par").read_text().splitlines()
assert len(SOURCE[1].split())>=9

def run(label,token,field=5,global_k=None):
    with tempfile.TemporaryDirectory(prefix="pr236-anmt-parser-") as td:
        d=Path(td)
        lines=SOURCE[:]
        if token is not None:
            p=lines[1].split()
            p[field]=token # mutate a specified field in the first KDPG spring
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
bending_k = {global_k if global_k is not None else "50.0"}
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
    for label,value in [("VALID_DECIMAL_EXPONENT","-4.9e-1"),("VALID_SIGNED_DECIMAL","+0.5")]:
        rc,log=run(label,value)
        if rc != 0:
            raise AssertionError(f"{label}: valid ANMT parameter was rejected: {log[-400:]}")
        print(f"VALID_ACCEPTED {label}",flush=True)

    invalid_cases=[
        ("NONFINITE_COSINE", "nan", 5, None),
        ("INFINITE_COSINE", "inf", 5, None),
        ("NEGATIVE_INFINITY_COSINE", "-inf", 5, None),
        ("COSINE_TRAILING_GARBAGE", "0.5junk", 5, None),
        ("COSINE_OVERFLOW", "1e999", 5, None),
        ("NONFINITE_SPRING_R0", "nan", 2, None),
        ("INFINITE_SPRING_STIFFNESS", "inf", 4, None),
        ("SPRING_R0_TRAILING_GARBAGE", "0.5junk", 2, None),
        ("GLOBAL_BENDING_NAN", None, 5, "nan"),
        ("GLOBAL_BENDING_INF", None, 5, "inf"),
        ("GLOBAL_BENDING_TRAILING_GARBAGE", None, 5, "50junk"),
    ]
    for label,value,field,global_k in invalid_cases:
        rc,log=run(label,value,field,global_k)
        if rc==0:
            raise AssertionError(f"{label}: invalid ANMT parameter was accepted")
        print(f"INVALID_REJECTED {label}: {log[-250:].replace(chr(10),' | ')}",flush=True)
    print("PASS: all valid decimal controls accepted and invalid ANMT tokens rejected",flush=True)

if __name__=="__main__":main()
