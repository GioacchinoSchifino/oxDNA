#!/usr/bin/env python3
"""Regression test: ANM energy decomposition and interchain springs.
Quantifies whether potential_energy (direct pair_interaction) equals the
sum of potential_energy split terms in DNANM vs DNACT after perturbing
one protein orientation. No physics/source code is changed.
"""
import argparse
import math
import subprocess
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/"examples"/"DNANM"/"KDPG"

def rotated_configuration(out):
    rows=(DATA/"s1.dat").read_text().splitlines()
    # first 3 lines: time, box, E; protein bead index 1 is row 4.
    i=4
    fields=rows[i].split()
    assert len(fields)>=15, (len(fields),rows[i][:120])
    v=[float(x) for x in fields]
    a1=v[3:6]
    a3=v[6:9]
    a2=[a3[1]*a1[2]-a3[2]*a1[1],
        a3[2]*a1[0]-a3[0]*a1[2],
        a3[0]*a1[1]-a3[1]*a1[0]]
    theta=0.87
    q=[math.cos(theta)*x+math.sin(theta)*y for x,y in zip(a1,a2)]
    v[3:6]=q
    rows[i]=" ".join(f"{x:.12g}" for x in v)
    out.write_text("\n".join(rows)+"\n")

def read_row(path):
    raw=path.read_text()
    candidates=[]
    for line in raw.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):continue
        try:
            row=[float(v) for v in line.split()]
            if len(row)>=4 and all(math.isfinite(v) for v in row):
                candidates.append(row)
        except ValueError:pass
    if not candidates:raise RuntimeError("No numeric split energy row: "+raw[-1200:])
    return candidates[0]

def one(binary, typ):
    with tempfile.TemporaryDirectory(prefix="erik-pr236-energy-") as tmp:
        d=Path(tmp)
        rotated_configuration(d/"rotated.dat")
        par=DATA/("kdpg_act.par" if typ=="DNACT" else "kdpg_anm.par")
        (d/"input").write_text(f"""backend = CPU
interaction_type = {typ}
sim_type = MD
steps = 1
T = 300K
dt = 0.0001
thermostat = no
refresh_vel = 0
seed = 1
restart_step_counter = 1
time_scale = linear
verlet_skin = 0.05
salt_concentration = 1.0
parfile = {par}
topology = {DATA/'generated.top'}
conf_file = rotated.dat
trajectory_file = trajectory.dat
lastconf_file = last_conf.dat
log_file = run.log
energy_file = energy.dat
no_stdout_energy = 1
print_energy_every = 1
print_conf_interval = 1
bending_k = 50.0
torsion_k = 50.0
data_output_1 = {{
 name = split.dat
 print_every = 1
 col_1 = {{
   type = potential_energy
   split = false
 }}
 col_2 = {{
   type = potential_energy
   split = true
 }}
}}
""")
        p=subprocess.run([str(binary),"input"],cwd=d,capture_output=True,text=True,timeout=180)
        if p.returncode:
            log=(d/"run.log").read_text() if (d/"run.log").exists() else "<no log>"
            raise RuntimeError(f"{typ} failed: {log[-3500:]} {p.stdout[-500:]} {p.stderr[-500:]}")
        row=read_row(d/"split.dat")
        direct=row[0]
        terms=row[1:]
        sumterms=sum(terms)
        delta=sumterms-direct
        print(f"{typ}: potential_energy/N={direct:.12g}, sum of terms/N={sumterms:.12g}, delta={delta:+.12g}, terms={terms}",flush=True)
        return direct,sumterms,delta


def test_interchain_index_boundary(binary):
    """An interchain spring between consecutive GLOBAL indices is not a
    protein backbone bond and must not require bending/torsion cosines.
    """
    with tempfile.TemporaryDirectory(prefix="erik-pr236-interchain-") as temp:
        d=Path(temp)
        (d/"system.top").write_text(
            "3 3 1 2 1 5->3\n"
            "A type=DNA circular=false\n"
            "K type=peptide circular=false\n"
            "E type=peptide circular=false\n")
        (d/"springs.par").write_text("2\n1 2 0.60 s 1.0\n")
        lines=["t = 0","b = 20 20 20","E = 0 0 0"]
        for pos in [(12,12,12),(5,5,5),(5.6,5,5)]:
            vals=(*pos,1,0,0,0,0,1,0,0,0,0,0,0)
            lines.append(" ".join(str(v) for v in vals))
        (d/"init.dat").write_text("\n".join(lines)+"\n")
        def run(typ):
            (d/"input").write_text(f"""backend = CPU
interaction_type = {typ}
sim_type = MD
steps = 0
T = 300K
dt = 0.001
thermostat = no
refresh_vel = 1
seed = 1
restart_step_counter = 1
time_scale = linear
verlet_skin = 0.20
salt_concentration = 1.0
parfile = springs.par
topology = system.top
conf_file = init.dat
trajectory_file = trajectory.dat
lastconf_file = lastconf.dat
log_file = run.log
energy_file = energy.dat
no_stdout_energy = 1
print_energy_every = 1
print_conf_interval = 1
bending_k = 50
torsion_k = 50
""")
            p=subprocess.run([str(binary),"input"],cwd=d,capture_output=True,text=True,timeout=60)
            log=(d/"run.log").read_text() if (d/"run.log").exists() else "<no-log>"
            return p.returncode,log
        a,la=run("DNANM")
        if a!=0:
            raise RuntimeError("Control DNANM incorrectly rejected cross-chain spring: "+la[-1300:])
        b,lb=run("DNACT")
        if b==0:
            print("DNACT accepts cross-chain adjacent-index spring without torsion cosines",flush=True)
        else:
            raise AssertionError("DNACT rejected a valid cross-chain spring: "+lb[-1600:])

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--binary",type=Path,required=True)
    binary=p.parse_args().binary.resolve()
    dn=one(binary,"DNANM")
    ct=one(binary,"DNACT")
    print("OBSERVED: DNANM delta=",dn[2],"DNACT delta=",ct[2],flush=True)
    test_interchain_index_boundary(binary)
    if max(abs(dn[2]), abs(ct[2])) > 1e-6:
        raise AssertionError(f"Potential energy disagrees with split ANM terms: DNANM {dn[2]}, DNACT {ct[2]}")
    print("PASS: pair energies agree with split terms and interchain spring is valid",flush=True)
if __name__=="__main__":main()
