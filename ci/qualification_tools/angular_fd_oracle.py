#!/usr/bin/env python3
"""Independent DNACT angular energy oracle and tiny executable checks.

Uses only the scalar four-cosine energy definition; forces, lab torques and
affine virial are obtained by finite differences, not copied force formulae.
CPU observables are native. CUDA force/torque are inferred from a one-step
impulse; host energy and native CUDA energy are reported separately.
No third-party Python packages required. Does not modify project sources.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess


def add(a, b): return tuple(x+y for x,y in zip(a,b))
def sub(a, b): return tuple(x-y for x,y in zip(a,b))
def mul(a, s): return tuple(x*s for x in a)
def dot(a, b): return sum(x*y for x,y in zip(a,b))
def cross(a,b): return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def norm(a): return math.sqrt(dot(a,a))
def unit(a): return mul(a,1/norm(a))
def axis(i): return tuple(float(i==j) for j in range(3))


def rotate(v, k, theta):
    k=unit(k); c=math.cos(theta); s=math.sin(theta)
    return add(add(mul(v,c),mul(cross(k,v),s)),mul(k,dot(k,v)*(1-c)))


def frame(k, theta):
    return (rotate((1.,0.,0.),k,theta),rotate((0.,0.,1.),k,theta))


TARGET=(.2,-.35,.15,-.25)
FRAMES=(frame((1.,2.,-1.),.47),frame((-2.,1.,3.),-.63))
POSITIONS=((4.,5.,6.),(5.1,5.6,5.55))
BOX=20.
COMPONENTS=((0,0),(1,1),(2,2),(0,1),(0,2),(1,2))


def minimum_image(r, box=BOX):
    return tuple(x-box*math.floor(x/box+.5) for x in r)


def energy(positions, frames, kb,kt):
    u=unit(minimum_image(sub(positions[1],positions[0])))
    a1,a3=frames[0]; b1,b3=frames[1]
    cs=(dot(u,a1),-dot(u,b1),dot(a1,b1),dot(a3,b3))
    ds=tuple(x-y for x,y in zip(cs,TARGET))
    return .5*(kb*(ds[0]**2+ds[1]**2)+kt*(ds[2]**2+ds[3]**2))


def derivative(f,h): return (f(h)-f(-h))/(2*h)
def richardson(f,h): return (4*derivative(f,h/2)-derivative(f,h))/3


def fd_response(positions,frames,kb,kt,h=1e-4):
    forces=[]; torques=[]
    for bead in range(2):
        fs=[]; ts=[]
        for j in range(3):
            def moved(e):
                pp=list(positions); pp[bead]=add(pp[bead],mul(axis(j),e))
                return energy(pp,frames,kb,kt)
            def turned(e):
                ff=list(frames); ff[bead]=tuple(rotate(v,axis(j),e) for v in ff[bead])
                return energy(positions,ff,kb,kt)
            fs.append(-richardson(moved,h)); ts.append(-richardson(turned,h))
        forces.append(tuple(fs)); torques.append(tuple(ts))
    r=minimum_image(sub(positions[1],positions[0])); virial=[]
    for i,j in COMPONENTS:
        def strained(e):
            pp=(positions[0],add(positions[0],add(r,mul(axis(j),e*r[i]))))
            return energy(pp,frames,kb,kt)
        virial.append(-richardson(strained,h))
    return {"energy":energy(positions,frames,kb,kt),"forces":forces,"torques_lab":torques,"virial":virial}


def all_finite(x):
    if isinstance(x,dict): return all(all_finite(v) for v in x.values())
    if isinstance(x,(list,tuple)): return all(all_finite(v) for v in x)
    return math.isfinite(x)


def check_close(label, actual, expected, atol,rtol):
    if len(actual)!=len(expected): raise RuntimeError(label+": shape mismatch")
    err=0.
    for a,e in zip(actual,expected):
        if isinstance(a,(list,tuple)):
            err=max(err,check_close(label,a,e,atol,rtol)); continue
        if not math.isfinite(a) or not math.isfinite(e): raise RuntimeError(label+": nonfinite")
        err=max(err,abs(a-e))
        if abs(a-e)>atol+rtol*abs(e):
            raise RuntimeError(f"{label}: actual={a:.16g}, expected={e:.16g}, error={a-e:.6g}")
    return err


def self_test():
    results=[]
    for kb,kt in ((45.,32.),(50.,50.),(45.,0.),(0.,32.),(0.,0.)):
        ref=fd_response(POSITIONS,FRAMES,kb,kt)
        if not all_finite(ref): raise RuntimeError("Oracle nonfinite")
        conv=check_close("FD convergence",fd_response(POSITIONS,FRAMES,kb,kt,h=5e-5)["forces"],ref["forces"],1e-7,1e-8)
        check_close("Newton III",add(*ref["forces"]),(0.,0.,0.),1e-7,0.)
        r=sub(POSITIONS[1],POSITIONS[0])
        angular_momentum=add(add(*ref["torques_lab"]),cross(r,ref["forces"][1]))
        check_close("Angular momentum",angular_momentum,(0.,0.,0.),2e-7,0.)
        check_close("Affine virial",ref["virial"],tuple(r[i]*ref["forces"][1][j] for i,j in COMPONENTS),1e-7,1e-8)
        check_close("Scale invariant trace",(sum(ref["virial"][:3]),),(0.,),1e-7,0.)
        periodic=(POSITIONS[0],add(POSITIONS[1],(BOX,0.,-BOX)))
        check_close("PBC translation",(energy(periodic,FRAMES,kb,kt),),(ref["energy"],),1e-10,1e-12)
        k=unit((2.,-3.,1.)); theta=.31
        rr=tuple(rotate(p,k,theta) for p in POSITIONS)
        ff=tuple(tuple(rotate(v,k,theta) for v in f) for f in FRAMES)
        check_close("Rigid rotation energy",(energy(rr,ff,kb,kt),),(ref["energy"],),1e-10,1e-12)
        results.append({"kb":kb,"kt":kt,"fd_convergence_max_abs":conv,**ref})
    return {"status":"PASS","scope":"scalar oracle only; not executable or CUDA qualification","cases":results}


def configuration(positions,frames):
    lines=["t = 0",f"b = {BOX} {BOX} {BOX}","E = 0 0 0"]
    for p,(a1,a3) in zip(positions,frames):
        # Nonzero tiny angular momentum avoids the zero-L GPU normalisation singularity.
        vals=(*p,*a1,*a3,0.,0.,0.,1e-8,2e-8,-1e-8)
        lines.append(" ".join(f"{x:.17g}" for x in vals))
    return "\n".join(lines)+"\n"


def input_text(backend,kb,kt,dt):
    cuda=""
    if backend=="CUDA":
        cuda="backend_precision = mixed\nCUDA_list = verlet\nCUDA_sort_every = 0\nuse_edge = 1\nedge_n_forces = 1\nCUDA_print_energy = true\nCUDA_update_stress_tensor_every = 1\nCUDA_update_particle_stress_tensor = true\n"
    # CUDA force/torque is measured from the native impulse below, not from
    # BaseParticle host fields. Keep the CPU observable and output IDs contiguous.
    force_output=""
    first_output=1
    if backend=="CPU":
        first_output=2
        force_output="""data_output_1 = {
 name = forces.dat
 print_every = 2
 col_1 = {
  type = force_and_torque
  particle = all
  lab_frame = true
 }
}
"""
    return f"""backend = {backend}
{cuda}interaction_type = DNACT
T = 300K
salt_concentration = 1.0
sim_type = MD
steps = 1
dt = {dt}
thermostat = no
refresh_vel = 0
restart_step_counter = 1
time_scale = linear
verlet_skin = 0.25
max_density_multiplier = 100
bending_k = {kb}
torsion_k = {kt}
parfile = springs.par
topology = system.top
conf_file = init.dat
trajectory_file = trajectory.dat
lastconf_file = last.dat
energy_file = native_energy.dat
log_file = log.dat
no_stdout_energy = 1
print_conf_interval = 2
print_energy_every = 2
{force_output}data_output_{first_output} = {{
 name = host_energy.dat
 print_every = 2
 col_1 = {{
  type = potential_energy
  precision = 17
 }}
}}
data_output_{first_output+1} = {{
 name = pressure.dat
 print_every = 2
 col_1 = {{
  type = pressure
  custom_stress_tensor = true
  stress_tensor = true
  PV_only = true
 }}
}}
data_output_{first_output+2} = {{
 name = native_backend_trace.dat
 print_every = 1
 col_1 = {{
  type = step
 }}
 col_2 = {{
  type = backend_info
 }}
}}
"""


def numeric_rows(path,ncols):
    rows=[]
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"): continue
        vals=[float(x) for x in line.split()]
        if len(vals)!=ncols: raise RuntimeError(f"Unexpected shape in {path}: {line}")
        if not all_finite(vals): raise RuntimeError(f"Nonfinite data in {path}")
        rows.append(vals)
    if not rows: raise RuntimeError(f"Empty file: {path}")
    return rows


def native_case(binary,backend,out,kb,kt,positions=POSITIONS,frames=FRAMES,dt=1e-5):
    out.mkdir(parents=True,exist_ok=False)
    (out/"system.top").write_text("2 1 0 2 0\n-1 K -1 1\n-1 K 0 -1\n")
    (out/"init.dat").write_text(configuration(positions,frames))
    r=norm(minimum_image(sub(positions[1],positions[0])))
    (out/"springs.par").write_text(f"2\n0 1 {r:.17g} s 0 "+" ".join(map(str,TARGET))+"\n")
    (out/"input").write_text(input_text(backend,kb,kt,dt))
    env=os.environ.copy()
    if backend=="CUDA": env["OXDNA_EXPERIMENTAL_DNACT_CUDA"]="RC2_GLOBAL_ANGULAR_RESEARCH_ONLY"
    p=subprocess.run([str(binary),"input"],cwd=out,env=env,capture_output=True,text=True,timeout=90)
    (out/"process.stdout").write_text(p.stdout); (out/"process.stderr").write_text(p.stderr)
    log=(out/"log.dat").read_text() if (out/"log.dat").exists() else ""
    if p.returncode or "END OF THE SIMULATION, everything went OK!" not in log:
        raise RuntimeError(f"{out}: executable failed {p.returncode}: {p.stdout[-1000:]} {p.stderr[-1000:]} {log[-1500:]}")
    if backend=="CUDA" and "Running on device 0" not in log: raise RuntimeError("No native CUDA device confirmation")
    state=[list(map(float,line.split())) for line in (out/"last.dat").read_text().splitlines()[3:] if line.strip()]
    if len(state)!=2 or any(len(x)!=15 for x in state) or not all_finite(state): raise RuntimeError("Invalid final state")
    host_e=numeric_rows(out/"host_energy.dat",1)[0][0]*2
    stress=numeric_rows(out/"pressure.dat",7)[0][1:]
    if backend=="CPU":
        rows=numeric_rows(out/"forces.dat",7)[:2]
        if len(rows)!=2 or {int(x[0]) for x in rows}!={0,1}: raise RuntimeError("Missing native CPU force rows")
        rows=sorted(rows,key=lambda x:x[0]); forces=[x[1:4] for x in rows]; torques=[x[4:7] for x in rows]
    else:
        forces=[mul(x[9:12],1/dt) for x in state]
        torques=[]
        for vals,(a1,a3) in zip(state,frames):
            body=mul(sub(vals[12:15],(1e-8,2e-8,-1e-8)),1/dt)
            a2=cross(a3,a1)
            torques.append(add(add(mul(a1,body[0]),mul(a2,body[1])),mul(a3,body[2])))
    result={"energy_host_total":host_e,"forces":forces,"torques_lab":torques,"virial":stress,"host_energy_rows":numeric_rows(out/"native_energy.dat",4)}
    if backend=="CUDA":
        trace=(out/"native_backend_trace.dat").read_text()
        matches=re.findall(r"(?m)^\s*(\d+)\s+CUDA_energy:\s*([-+0-9.eE]+)\s*$",trace)
        if len(matches)!=1 or int(matches[0][0])!=1: raise RuntimeError("Native CUDA energy missing or not explicitly aligned to final step 1: "+trace)
        native_e=float(matches[0][1])*2
        final_positions=tuple(tuple(x[:3]) for x in state)
        final_frames=tuple((tuple(x[3:6]),tuple(x[6:9])) for x in state)
        final_reference=energy(final_positions,final_frames,kb,kt)
        result.update(native_cuda_energy_total=native_e,native_cuda_energy_step=1,final_energy_scalar_reference=final_reference,
                      native_cuda_energy_error=check_close("Native CUDA energy at final step 1",(native_e,),(final_reference,),2e-4,2e-5))
    return result


def run_executable(binary,backend,out):
    binary=Path(binary).resolve()
    if not binary.is_file(): raise RuntimeError("Missing binary")
    cases=[]
    for j,(kb,kt) in enumerate(((45.,32.),(50.,50.),(45.,0.),(0.,32.),(0.,0.))):
        actual=native_case(binary,backend,out/f"case_{j}_{kb}_{kt}",kb,kt)
        ref=fd_response(POSITIONS,FRAMES,kb,kt)
        atol,rtol=(2e-5,8e-6) if backend=="CPU" else (2e-3,2e-4)
        errors={"energy_host":check_close("host energy",(actual["energy_host_total"],),(ref["energy"],),1e-9 if backend=="CPU" else 2e-4,2e-12 if backend=="CPU" else 2e-5),
                "force":check_close("native force vs independent FD",actual["forces"],ref["forces"],atol,rtol),
                "torque":check_close("native torque vs independent FD",actual["torques_lab"],ref["torques_lab"],atol,rtol),
                "virial":check_close("native virial vs independent affine FD",actual["virial"],ref["virial"],2e-6 if backend=="CPU" else 2e-3,1e-6 if backend=="CPU" else 2e-4)}
        cases.append({"kb":kb,"kt":kt,"errors_max_abs":errors,"native":actual,"reference_fd":ref})
    # Native PBC equivalence, away from nearest-image branch boundaries.
    periodic=(POSITIONS[0],add(POSITIONS[1],(BOX,0.,-BOX)))
    shifted=native_case(binary,backend,out/"pbc_shifted",45.,32.,positions=periodic)
    check_close("native PBC force",shifted["forces"],cases[0]["native"]["forces"],atol,rtol)
    check_close("native PBC torque",shifted["torques_lab"],cases[0]["native"]["torques_lab"],atol,rtol)
    return {"status":"PASS","backend":backend,"binary":str(binary),"binary_sha256":hashlib.sha256(binary.read_bytes()).hexdigest(),"cases":cases,"pbc_shifted":shifted,
            "scope":"CPU native initial forces; CUDA force/torque one-step impulse approximation. Host energy is CPU recomputation; native CUDA energy is backend_info explicitly at final step 1 and checked against scalar energy at final configuration. No NVE/NVT qualification."}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--binary"); p.add_argument("--backend",choices=("CPU","CUDA"),default="CPU"); p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    result={"oracle_self_test":self_test()}
    try:
        if a.binary: result["executable"]=run_executable(a.binary,a.backend,a.out)
    except Exception as e:
        result["status"]="FAIL"; result["error"]=str(e)
        (a.out/"result.json").write_text(json.dumps(result,indent=2)+"\n"); raise
    result["status"]="PASS"; (a.out/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"status":result["status"],"result":str(a.out/"result.json"),"executable_tested":bool(a.binary)}))


if __name__=="__main__": main()
