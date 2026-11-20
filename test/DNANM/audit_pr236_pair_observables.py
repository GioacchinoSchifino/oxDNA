#!/usr/bin/env python3
"""Regression: PairForce angular torque and PairEnergy column labels.

The same pinned KDPG DNACT input runs twice:
(A) unchanged PairForce.cpp calling pair_interaction(q,p)
(B) ONLY its argument order is canonicalized in the runner copy.
The physics engine and all other files are identical.
"""
import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path

REPO=Path(__file__).resolve().parents[2]
DATA=REPO/"examples"/"DNANM"/"KDPG"

def rotate(source, dest):
    lines=source.read_text().splitlines()
    vals=[float(v) for v in lines[3].split()]
    a1=vals[3:6]
    a3=vals[6:9]
    a2=(a3[1]*a1[2]-a3[2]*a1[1],a3[2]*a1[0]-a3[0]*a1[2],a3[0]*a1[1]-a3[1]*a1[0])
    t=0.87
    vals[3:6]=[math.cos(t)*x+math.sin(t)*y for x,y in zip(a1,a2)]
    lines[3]=" ".join("%.15g"%v for v in vals)
    dest.write_text("\n".join(lines)+"\n")

def run(binary, pairenergy=False):
    with tempfile.TemporaryDirectory(prefix="pr236-pair-observable-") as w:
        d=Path(w)
        rotate(DATA/"s1.dat",d/"rotated.dat")
        extra=""
        if pairenergy:
            extra="""data_output_2 = {
 name = pair_energy.dat
 print_every = 1
 col_1 = {
  type = pair_energy
  particle1_id = 0
  particle2_id = 1
  print_header = true
 }
}
"""
        (d/"input").write_text(f"""backend = CPU
interaction_type = DNACT
sim_type = MD
T = 300K
dt = 0.0001
thermostat = no
refresh_vel = 0
seed = 1729
restart_step_counter = 1
time_scale = linear
verlet_skin = 0.15
salt_concentration = 1.0
parfile = {DATA/'kdpg_act.par'}
topology = {DATA/'generated.top'}
conf_file = rotated.dat
trajectory_file = trajectory.dat
lastconf_file = lastconf.dat
log_file = log.dat
energy_file = energy.dat
no_stdout_energy = 1
print_conf_interval = 1
print_energy_every = 1
steps = 1
bending_k = 50.0
torsion_k = 50.0
data_output_1 = {{
 name = pair_force.dat
 print_every = 1
 col_1 = {{
  type = pair_force
  particle_id = 0
 }}
}}
{extra}""")
        p=subprocess.run([str(binary),"input"],cwd=d,capture_output=True,text=True,timeout=100)
        log=(d/"log.dat").read_text(errors="replace") if (d/"log.dat").exists() else "<no log>"
        if p.returncode:
            raise RuntimeError(f"DNACT observable subprocess exited {p.returncode}: "+log[-2300:])
        path=d/"pair_force.dat"
        if not path.is_file():raise RuntimeError("PairForce output absent")
        wanted={}
        for line in path.read_text().splitlines():
            a=line.split()
            if len(a)!=8:continue
            try:
                i,j=int(a[0]),int(a[1])
                values=[float(x) for x in a[2:]]
            except ValueError:continue
            if {i,j}=={0,1} and i not in wanted: wanted[i]=values
        if set(wanted)!={0,1}:
            raise RuntimeError(f"Missing selected bonded 0-1 pair in PairForce output: keys {set(wanted)}")
        o={"pair_0_1":wanted}
        if pairenergy:
            p=d/"pair_energy.dat"
            lines=p.read_text().splitlines()
            hdr=next((line for line in lines if line.startswith("#id1")),None)
            row=next((line for line in lines if line.strip() and not line.startswith("#")),None)
            if not hdr or not row:raise RuntimeError("PairEnergy lacks header or numerical row")
            labels=hdr.split(", t =")[0].split()
            numeric=row.split()
            o.update({"header":hdr,"pair_energy_row":row,"header_columns":len(labels),"data_columns":len(numeric)})
        return o

def compare(a,b):
    va=a["pair_0_1"];vb=b["pair_0_1"]
    f0=[float(vb["0"][i])-float(va["0"][i]) for i in range(6)]
    f1=[float(vb["1"][i])-float(va["1"][i]) for i in range(6)]
    n0=math.sqrt(sum(x*x for x in f0))
    n1=math.sqrt(sum(x*x for x in f1))
    tor0=math.sqrt(sum(x*x for x in f0[3:]))
    tor1=math.sqrt(sum(x*x for x in f1[3:]))
    print("DIFF PairForce patched-minus-original bead0:",f0,flush=True)
    print("DIFF PairForce patched-minus-original bead1:",f1,flush=True)
    print(f"DIFF norms total={n0:.10g}/{n1:.10g}, torque={tor0:.10g}/{tor1:.10g}",flush=True)
    if max(tor0,tor1) < 1e-4:
        raise AssertionError("No measurable angular torque difference: check fixture or call path")
    print("CONFIRMED: original PairForce misses angular torque from bonded ANMT path",flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--binary",type=Path)
    p.add_argument("--output",type=Path)
    p.add_argument("--pairenergy",action="store_true")
    p.add_argument("--compare",nargs=2,type=Path,metavar=("ORIGINAL","PATCHED"))
    args=p.parse_args()
    if args.compare:
        compare(*[json.loads(f.read_text()) for f in args.compare])
        return
    if args.binary is None or args.output is None:
        p.error("Provide --binary and --output, or --compare FILE FILE")
    x=run(args.binary.resolve(),args.pairenergy)
    args.output.write_text(json.dumps(x,indent=2)+"\n")
    torques=[abs(v) for vec in x["pair_0_1"].values() for v in vec[3:]]
    if not torques or max(torques)<1e-6:
        raise AssertionError("Bonded ANMT PairForce omitted expected nonzero angular torque")
    print("PASS: PairForce includes a nonzero angular torque:",x["pair_0_1"],flush=True)
    if args.pairenergy:
        print("PairEnergy header:",x["header"],flush=True)
        print("PairEnergy numerical row:",x["pair_energy_row"],flush=True)
        print(f"PAIR ENERGY HEADER: {x['header_columns']} labels vs {x['data_columns']} numerical columns",flush=True)
        if x["header_columns"] != x["data_columns"]:
            raise AssertionError("ANM PairEnergy header columns do not match the output data")
        print("PASS: PairEnergy header columns agree with the numerical data",flush=True)

if __name__=="__main__":main()
