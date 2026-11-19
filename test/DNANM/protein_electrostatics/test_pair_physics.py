#!/usr/bin/env python3
"""Independent energy/force/torque checks for the new DNANM screened
electrostatics on the #236 architecture, using an isolated four-particle
DNA/protein configuration. Excludes the pre-existing PairEnergy observable.
"""
import argparse
import math
from pathlib import Path
import subprocess
import tempfile

N = 4
DNA = (5.0, 5.0, 5.0)
BACK = (-0.3400, 0.3408, 0.0)
P1 = (4.6600, 5.3408, 6.0200)
P2 = (12.0, 12.0, 12.0)
P3 = (5.4600, 5.3408, 6.0200)
LAMBDA = 0.3616455
STRENGTH = 0.0543
QDNA = 0.815
TOP = "4 4 1 3 1\n1 A -1 -1\n-1 K -1 -1\n-2 K -1 -1\n-3 K -1 -1\n"

def norm(v):
    return math.sqrt(sum(t*t for t in v))

def sub(a,b):
    return tuple(x-y for x,y in zip(a,b))

def vector(v, scalar):
    return tuple(scalar*x for x in v)

def dh(qproduct, rvec):
    r = norm(rvec)
    if not r: raise ValueError("zero separation")
    rc = 4.5 * LAMBDA
    if r >= rc: return 0.0, (0.0, 0.0, 0.0)
    # The configurations below lie on the pure exponential branch r<3 lambda.
    if not r < 3*LAMBDA: raise ValueError("reference fixture is outside pure-DH branch")
    k = 1.0 / LAMBDA
    e = math.exp(-k*r)
    U = qproduct * STRENGTH * e / r
    F = qproduct * STRENGTH * e * (k / r + 1/(r*r))
    return U, vector(rvec, F/r)

def cross(a,b):
    return (a[1]*b[2]-a[2]*b[1],
            a[2]*b[0]-a[0]*b[2],
            a[0]*b[1]-a[1]*b[0])

def conf_text(p1):
    particles = [(DNA, (1.,0.,0.), (0.,0.,1.)),
                 (p1,  (1.,0.,0.), (0.,0.,1.)),
                 (P2,  (1.,0.,0.), (0.,0.,1.)),
                 (P3,  (1.,0.,0.), (0.,0.,1.))]
    rows=["t = 0", "b = 20 20 20", "E = 0 0 0"]
    for pos, a1, a3 in particles:
        rows.append(" ".join(f"{v:.15g}" for v in (*pos,*a1,*a3,0,0,0,0,0,0)))
    return "\n".join(rows)+"\n"

def first_energy(path):
    if not path.exists(): raise RuntimeError("No energy output")
    for line in path.read_text().splitlines():
        try:
            nums=[float(x) for x in line.split()]
            if len(nums)>=2 and all(math.isfinite(x) for x in nums):
                return nums[1] * N # oxDNA PotentialEnergy reports U/N
        except ValueError:
            pass
    raise RuntimeError("No finite potential energy output")

def first_pair_forces(path, pair):
    observed={}
    if not path.exists(): return observed
    for line in path.read_text().splitlines():
        cols=line.split()
        if len(cols)!=8: continue
        try:
            p,q=int(cols[0]),int(cols[1])
            vals=tuple(float(c) for c in cols[2:])
        except ValueError: continue
        if {p,q} == set(pair) and p not in observed:
            observed[p]=(vals[:3],vals[3:])
        if len(observed)==2: break
    return observed

def run(binary, *, es, charges=(1.,0.,0.), groups=(0,1,2),
        p1=P1, pdscale=1., ppscale=1., half=False, bonded=False,
        malformed=None):
    with tempfile.TemporaryDirectory(prefix="oxdna-es-physics-") as tmp:
        d=Path(tmp)
        (d/"system.top").write_text(TOP)
        (d/"init.dat").write_text(conf_text(p1))
        (d/"springs.par").write_text("3\n1 3 0.8 s 1.0\n" if bonded else "3\n")
        lines=["# DNANM_ES_CHARGES 1",
               "# index_space = protein_local", "# index_base = 0",
               "# n_protein_beads = 3",
               f"# n_groups = {len(set(groups))}"]
        for i, (strand, q, g) in enumerate(zip((-1,-2,-3),charges,groups)):
            qtext=str(q) if i!=0 or malformed is None else malformed
            lines.append(f"{i} {strand} K {qtext} {g}")
        (d/"charges.dat").write_text("\n".join(lines)+"\n")
        inp=f"""backend = CPU
interaction_type = DNANM
sim_type = MD
T = 300K
dt = 0.002
thermostat = no
refresh_vel = 1
seed = 314159
restart_step_counter = 1
time_scale = linear
verlet_skin = 0.25
salt_concentration = 1.0
dh_half_charged_ends = {'true' if half else 'false'}
ensemble = NVT
external_forces = 0
max_io = 100
parfile = springs.par
topology = system.top
conf_file = init.dat
trajectory_file = trajectory.dat
lastconf_file = last_conf.dat
energy_file = energy.dat
log_file = log.dat
no_stdout_energy = 1
steps = 1
print_conf_interval = 1
print_energy_every = 1
protein_electrostatics = {'true' if es else 'false'}
protein_charge_file = charges.dat
protein_charge_scale = 1.0
protein_dna_dh_scale = {pdscale}
protein_protein_dh_scale = {ppscale}
dna_effective_charge_e = 0.815
exclude_intragroup_electrostatics = true
data_output_1 = {{
 name = pair_force.dat
 print_every = 1
 col_1 = {{
  type = pair_force
  particle_id = 1
 }}
}}
"""
        (d/"input").write_text(inp)
        p=subprocess.run([str(binary), "input"], cwd=d,
                         text=True,capture_output=True,timeout=60)
        log=(d/"log.dat").read_text() if (d/"log.dat").exists() else "<no log>"
        if malformed is not None:
            if p.returncode == 0:
                raise AssertionError("Malformed protein charge accepted")
            return None
        if p.returncode:
            raise RuntimeError(f"oxDNA failed exit={p.returncode}\n{log[-2500:]}\n{p.stdout[-900:]}")
        return first_energy(d/"energy.dat"), first_pair_forces(d/"pair_force.dat",(0,1)), \
               first_pair_forces(d/"pair_force.dat",(1,3))

def close(x,y, *, abs_tol=3e-5, rel_tol=0.02, label="value"):
    if not math.isclose(x,y,rel_tol=rel_tol,abs_tol=abs_tol):
        raise AssertionError(f"{label}: expected {y:.12g} observed {x:.12g}")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--binary",required=True,type=Path)
    binary=p.parse_args().binary.resolve()

    off=run(binary,es=False,pdscale=1.,ppscale=0.)
    on=run(binary,es=True,pdscale=1.,ppscale=0.)
    expected,fp=dh(-1./QDNA,sub(P1,tuple(a+b for a,b in zip(DNA,BACK))))
    close(on[0]-off[0],expected,label="PD energy")
    if not all(pid in on[1] for pid in (0,1)):
        raise AssertionError("PairForce did not report charged PD pair")
    # PairForce returns total force; subtract ES-OFF to obtain the ES term.
    base=off[1]
    for pid in (0,1):
        if pid not in base: base[pid]=((0.,0.,0.),(0.,0.,0.))
    fpro=tuple(on[1][1][0][i]-base[1][0][i] for i in range(3))
    fnuc=tuple(on[1][0][0][i]-base[0][0][i] for i in range(3))
    for i in range(3):
        close(fpro[i],fp[i],label=f"PD force protein[{i}]",abs_tol=1e-4)
        close(fnuc[i],-fp[i],label=f"PD force DNA[{i}]",abs_tol=1e-4)
        close(fnuc[i]+fpro[i],0.,label="Newton third law",abs_tol=1e-4)
    expected_torque=cross(BACK,tuple(-v for v in fp))
    for i in range(3):
        value=on[1][0][1][i]-base[0][1][i]
        close(value,expected_torque[i],label=f"DNA BACK torque[{i}]",abs_tol=1e-4)
    print("PASS protein-DNA analytic energy, force, reaction force, nucleotide torque")

    half=run(binary,es=True,pdscale=1.,ppscale=0.,half=True)
    close(half[0]-off[0],expected/2,label="terminal DNA half-charge")
    print("PASS DNA terminal half-charge")

    off_pp=run(binary,es=False,charges=(1.,0.,-1.),pdscale=0.,ppscale=1.)
    on_pp=run(binary,es=True,charges=(1.,0.,-1.),pdscale=0.,ppscale=1.)
    u_pp,fpp=dh(-1./(QDNA*QDNA),sub(P3,P1))
    close(on_pp[0]-off_pp[0],u_pp,label="PP energy")
    if 3 not in on_pp[2]: raise AssertionError("PairForce did not report charged PP pair")
    basepp=off_pp[2].get(3,((0.,0.,0.),(0.,0.,0.)))
    for i in range(3):
        obs=on_pp[2][3][0][i]-basepp[0][i]
        close(obs,fpp[i],label=f"PP force[{i}]",abs_tol=1e-4)
    print("PASS protein-protein analytic energy and force")

    same=run(binary,es=True,charges=(1.,0.,-1.),groups=(0,0,0),pdscale=0.,ppscale=1.)
    close(same[0]-off_pp[0],0.,label="PP intragroup exclusion")
    off_bonded=run(binary,es=False,charges=(1.,0.,-1.),pdscale=0.,ppscale=1.,bonded=True)
    bonded=run(binary,es=True,charges=(1.,0.,-1.),pdscale=0.,ppscale=1.,bonded=True)
    close(bonded[0]-off_bonded[0],0.,label="PP spring-bonded exclusion")
    print("PASS intragroup and spring-bonded electrostatic exclusions")

    run(binary,es=True,malformed="nan")
    run(binary,es=True,malformed="1e9999")
    print("PASS fail-closed for NaN and overflowing charges")
    print("CPU ANALYTICAL PHYSICS PASS")

if __name__=="__main__":
    main()
