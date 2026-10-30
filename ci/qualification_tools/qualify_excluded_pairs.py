#!/usr/bin/env python3
"""Scalar-U FD oracles, one-step native forces, aligned device U and stress.
No project imports, job submission, compilation or source mutation.
"""
import argparse
from dataclasses import dataclass
import hashlib,json,math,os,re,subprocess,sys
from pathlib import Path

BOX,DT=20.,1e-6
COMP=((0,0),(1,1),(2,2),(0,1),(0,2),(1,2))
RAY=tuple(x/math.sqrt(29.) for x in (2.,-3.,4.))
BACK=(-.34,.3408,0.)
LAM,QDH=.3616455,.0543
RHIGH=3*LAM
RC=RHIGH*(RHIGH+3*LAM)/(RHIGH+LAM)
BDH=math.exp(-RHIGH/LAM)*QDH*(RHIGH+LAM)**2/(4*RHIGH**3*LAM**2)
def add(a,b):return tuple(x+y for x,y in zip(a,b))
def sub(a,b):return tuple(x-y for x,y in zip(a,b))
def mul(a,s):return tuple(x*s for x in a)
def norm(a):return math.sqrt(sum(x*x for x in a))
def mic(a):return tuple(x-BOX*math.floor(x/BOX+.5) for x in a)

@dataclass(frozen=True)
class Case:
    name:str
    r:float=.34
    pair:tuple=(0,1)
    csr:bool=False
    topology:str='reciprocal'
    pp:bool=False
    pd:bool=False
    sign:float=1.
    same_group:bool=False
    exclude_group:bool=True
    scale:float=1.
    periodic:bool=False
    dna_first:bool=False
    distinct_strands:bool=False
    nforces:int=1
    kb:float=0.
    kt:float=0.
    interaction:str='DNACT'
    sort:int=0
    per_bond:bool=False
    par_none:bool=False
    comment_par:bool=False
    precision:str='mixed'
    negative:str=''

CASES=(
 Case('EV_adjacent_CSR_backbone',csr=True),
 Case('EV_adjacent_CSR_no_backbone',csr=True,topology='none',distinct_strands=True),
 Case('EV_nonadjacent_backbone',pair=(0,2),csr=True),
 Case('EV_backbone_without_CSR'),
 Case('EV_quartic_smoothing',r=.351),
 Case('EV_above_cutoff',r=.36),
 Case('EV_duplicate_n3_n5',topology='duplicate'),
 Case('EV_periodic_oblique',periodic=True),
 Case('EV_DNA_first_offset',dna_first=True),
 Case('EV_nonbonded_edge4',csr=True,topology='none',nforces=4),
 Case('DH_screened',r=.8,pp=True),
 Case('DH_smoothing',r=1.3,pp=True),
 Case('DH_above_cutoff',r=1.7,pp=True),
 Case('DH_attraction',r=.8,pp=True,sign=-1.),
 Case('DH_CSR_exclusion',r=.8,pp=True,csr=True),
 Case('DH_group_exclusion',r=.8,pp=True,same_group=True),
 Case('DH_group_exclusion_disabled',r=.8,pp=True,same_group=True,exclude_group=False),
 Case('DH_scale',r=.8,pp=True,scale=.37),
 Case('DH_periodic_oblique',r=.8,pp=True,periodic=True),
 Case('DH_DNA_first_offset',r=.8,pp=True,dna_first=True),
 Case('EV_DH_composition_dedup',pp=True,topology='duplicate'),
 Case('global_angles_input_only',csr=True,topology='none',distinct_strands=True,kb=45.,kt=32.),
 Case('DNANM_unified_control',csr=True,topology='none',interaction='DNANM'),
 Case('C06_DNANM_parfile_none_EV',interaction='DNANM',par_none=True,topology='none'),
 Case('C06_DNANM_parfile_none_PD',interaction='DNANM',par_none=True,topology='none',dna_first=True,pd=True,r=.8),
 Case('C06_DNACT_parfile_none_reject',par_none=True,kb=45.,kt=32.,negative='validated ANM parameter file'),
 Case('C05_reject_parfile_comment',csr=True,comment_par=True,negative='Malformed or unsupported CUDA ANM parameter row'),
 Case('reject_nonreciprocal',topology='one_sided',negative='Nonreciprocal DNACT CUDA'),
 Case('reject_self_link',topology='self',negative='Invalid DNACT CUDA backbone'),
 Case('reject_per_bond',csr=True,per_bond=True,kb=-1.,kt=-1.,negative='per-bond'),
 Case('reject_sort_zero_angles',sort=5,negative='CUDA_sort_every'),
 Case('reject_sort_nonzero_angles',sort=5,kb=45.,kt=32.,negative='CUDA_sort_every'),
 Case('reject_sort_negative',sort=-5,negative='CUDA_sort_every'),
 Case('reject_precision_double',precision='double',negative='mixed'),
)

def ev_energy(r):
    if r>=.352894:return 0.
    if r>.349:return 2*306484596.*(r-.352894)**4
    lj6=(.35/r)**6
    return 8*(lj6*lj6-lj6)
def dh_energy(r):
    if r>=RC:return 0.
    return QDH*math.exp(-r/LAM)/r if r<RHIGH else BDH*(r-RC)**2
def scalar(case,rv):
    if case.pd:return -dh_energy(norm(sub(rv,BACK)))
    qq=case.scale*case.sign if case.pp and not case.csr and not(case.exclude_group and case.same_group) else 0.
    return ev_energy(norm(rv))+qq*dh_energy(norm(rv))
def derivative(fn,h):
    return (4*(fn(h/2)-fn(-h/2))/h-(fn(h)-fn(-h))/(2*h))/3
def oracle(case,rv=None):
    rv=(add(BACK,mul(RAY,case.r)) if case.pd else mul(RAY,case.r)) if rv is None else rv
    forces=[];virial=[]
    for j in range(3):
        def moved(e):
            x=list(rv);x[j]+=e;return scalar(case,x)
        forces.append(derivative(moved,2e-6))
    for i,j in COMP:
        def strained(e):
            x=list(rv);x[j]+=e*rv[i];return scalar(case,x)
        virial.append(-derivative(strained,2e-6))
    return dict(energy_total=scalar(case,rv),force_first=forces,virial=virial)
def finite_tree(x):
    if isinstance(x,dict):return all(finite_tree(v) for v in x.values())
    if isinstance(x,(list,tuple)):return all(finite_tree(v) for v in x)
    return isinstance(x,(int,float)) and math.isfinite(x)
def compare(label,actual,expected,atol=2e-5,rtol=5e-4):
    if len(actual)!=len(expected):raise ValueError(label+': shape mismatch')
    errors=[]
    for a,e in zip(actual,expected):
        if isinstance(e,(tuple,list)):
            errors.append(compare(label,a,e,atol,rtol));continue
        if not math.isfinite(a) or not math.isfinite(e):raise ValueError(label+': nonfinite')
        errors.append(abs(a-e))
        if abs(a-e)>atol+rtol*abs(e):raise ValueError(f'{label}: actual={a:.16g};expected={e:.16g};error={a-e:.8g}')
    return max(errors,default=0.)

def fixture(case):
    center=(19.9,10.,10.) if case.periodic else (4.,5.,6.)
    other=tuple(x%BOX for x in add(center,mul(RAY,case.r)))
    positions=[(9.,3.,4.) if case.periodic else (9.,9.,9.) for _ in range(3)]
    a,b=case.pair;positions[a],positions[b]=center,other
    if case.pd:positions=[add((4.,5.,6.),add(BACK,mul(RAY,case.r))),(9.,9.,9.),(12.,8.,10.)]
    edges={(0,2),(1,2)} if (a,b)==(0,1) else {(0,1),(1,2)}
    if case.csr:edges={(a,b),(1,2) if (a,b)==(0,1) else (0,1)}
    links=[[-1,-1] for _ in range(3)]
    if case.topology in ('reciprocal','duplicate'):
        links[a][1],links[b][0]=b,a
        if case.topology=='duplicate':links[a]=[b,b];links[b]=[a,a]
    elif case.topology=='one_sided':links[a][1]=b
    elif case.topology=='self':links[a][0]=a
    offset=int(case.dna_first)
    strands=[-(i+1) if case.distinct_strands else -1 for i in range(3)]
    top=f'{3+offset} {(3 if case.distinct_strands else 1)+offset} {offset} 3 {offset}\n'
    if offset:top+='1 A -1 -1\n'
    for i in range(3):
        x,y=(j+offset if j>=0 else -1 for j in links[i])
        top+=f'{strands[i]} K {x} {y}\n'
    par='3\n'
    for i,j in sorted(edges):
        rv=mic(sub(positions[j],positions[i]));r=norm(rv)
        par+=f'{i+offset} {j+offset} {r:.17g} s 0'
        if case.interaction=='DNACT' and j-i==1:
            par+=f' {rv[0]/r:.17g} {-rv[0]/r:.17g} 1 1'
            if case.per_bond:par+=' 45 32'
        par+='\n'
    q=[0.,0.,0.];q[a],q[b]=1.,case.sign
    if case.pd:q=[1.,0.,0.]
    groups=[0,1,2]
    if case.same_group:groups[a],groups[b]=0,0;groups[({0,1,2}-{a,b}).pop()]=1
    charges=f'# DNANM_ES_CHARGES 1\n# index_space = protein_local\n# index_base = 0\n# charge_model = independent-oracle\n# charge_source = synthetic\n# n_protein_beads = 3\n# n_groups = {len(set(groups))}\n# columns = protein_index topology_strand residue charge_e group_id\n'
    for i in range(3):charges+=f'{i} {strands[i]} K {.815*q[i]:.17g} {groups[i]}\n'
    if offset:positions=[(4.,5.,6.) if case.pd else (14.,14.,14.)]+positions
    conf=f't = 0\nb = {BOX} {BOX} {BOX}\nE = 0 0 0\n'
    for pos in positions:conf+=' '.join(f'{x:.17g}' for x in (*pos,1.,0.,0.,0.,0.,1.,0.,0.,0.,1e-8,2e-8,-1e-8))+'\n'
    if case.comment_par:
        lines=par.splitlines();lines.insert(2,'# interspersed comment intentionally unsupported on CUDA');par='\n'.join(lines)+'\n'
    return dict(positions=positions,topology=top,par=par,charges=charges,conf=conf,
        first=0 if case.pd else a+offset,second=1 if case.pd else b+offset,N=3+offset)

def input_text(case,backend):
    cuda=''
    if backend=='CUDA':cuda=f'''backend_precision = {case.precision}
CUDA_list = verlet
CUDA_sort_every = {case.sort}
use_edge = 1
edge_n_forces = {case.nforces}
CUDA_print_energy = 1
CUDA_update_stress_tensor_every = 1
CUDA_update_particle_stress_tensor = true
'''
    angular=f'bending_k = {case.kb}\ntorsion_k = {case.kt}\n' if case.interaction=='DNACT' else ''
    text=f'''backend = {backend}
{cuda}interaction_type = {case.interaction}
topology = topology.top
parfile = {'none' if case.par_none else 'network.par'}
conf_file = initial.dat
salt_concentration = 1
T = 300K
sim_type = MD
steps = 1
dt = {DT}
thermostat = no
refresh_vel = 0
seed = 87611
{angular}verlet_skin = 0.2
max_density_multiplier = 100
protein_protein_electrostatics = {int(case.pp)}
protein_dna_electrostatics = {int(case.pd)}
protein_dna_dh_scale = 1
dh_half_charged_ends = false
protein_charge_file = charges.dat
protein_charge_scale = 1
protein_protein_dh_scale = {case.scale}
exclude_intragroup_electrostatics = {int(case.exclude_group)}
trajectory_file = trajectory.dat
energy_file = energy.dat
lastconf_file = last.dat
log_file = log.dat
no_stdout_energy = 1
print_conf_interval = 2
print_energy_every = 2
time_scale = linear
restart_step_counter = 1
'''
    outputs=(('host_energy.dat','type = potential_energy\nprecision = 17'),
             ('pressure.dat','type = pressure\ncustom_stress_tensor = true\nstress_tensor = true\nPV_only = true'),
             ('particle_stress.dat','type = particle_stress\nprint_coordinates = false'))
    # Native CUDA force is read from the velocity impulse, never this host
    # observable. Enumerating the selected outputs also prevents ID gaps.
    if backend=='CPU':
        outputs=(('forces.dat','type = force_and_torque\nparticle = all\nlab_frame = true'),)+outputs
    for i,(name,contents) in enumerate(outputs,1):
        text+=f'data_output_{i} = {{\n name = {name}\n print_every = 2\n col_1 = {{\n{contents}\n }}\n}}\n'
    return text+f'''data_output_{len(outputs)+1} = {{
 name = native_trace.dat
 print_every = 1
 col_1 = {{
  type = step
 }}
 col_2 = {{
  type = backend_info
 }}
}}
'''

def numeric_rows(path,ncols):
    rows=[]
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):continue
        values=[float(v) for v in line.split()]
        if len(values)!=ncols or not finite_tree(values):raise ValueError(str(path)+': malformed/nonfinite data')
        rows.append(values)
    if not rows:raise ValueError(str(path)+': no numeric records')
    return rows

def run_case(binary,case,backend,out):
    out.mkdir(parents=True,exist_ok=False);fx=fixture(case)
    for name,key in (('topology.top','topology'),('network.par','par'),('charges.dat','charges'),('initial.dat','conf')):
        (out/name).write_text(fx[key])
    (out/'input').write_text(input_text(case,backend))
    env=os.environ.copy()
    env.pop('OXDNA_EXPERIMENTAL_DNACT_CUDA',None)
    env.pop('OXDNA_EXPERIMENTAL_SORT_CUDA',None)
    if case.sort:env['OXDNA_EXPERIMENTAL_SORT_CUDA']='GATE46_RESEARCH_ONLY'
    proc=subprocess.run([str(binary),'input'],cwd=out,env=env,text=True,capture_output=True,timeout=45)
    (out/'stdout.txt').write_text(proc.stdout);(out/'stderr.txt').write_text(proc.stderr)
    log=(out/'log.dat').read_text() if (out/'log.dat').exists() else ''
    if case.negative:
        if proc.returncode!=1 or case.negative.lower() not in (proc.stdout+proc.stderr+log).lower():
            raise ValueError(f'Expected explicit rejection {case.negative!r}; exit={proc.returncode}')
        return dict(status='FAIL_CLOSED_PASS',exit_code=proc.returncode,matched_message=case.negative)
    if proc.returncode!=0 or 'END OF THE SIMULATION, everything went OK!' not in log:
        raise ValueError(f'Execution failed exit={proc.returncode}: {proc.stderr[-1000:]} {log[-1000:]}')
    if backend=='CUDA' and 'Running on device 0' not in log:raise ValueError('Native CUDA execution not established')
    n=fx['N']
    final=[list(map(float,v.split())) for v in (out/'last.dat').read_text().splitlines()[3:] if v.strip()]
    if len(final)!=n or any(len(v)!=15 for v in final) or not finite_tree(final):raise ValueError('Incomplete final state')
    ref=oracle(case)
    forces=[[0.,0.,0.] for _ in range(n)]
    forces[fx['first']]=ref['force_first'];forces[fx['second']]=mul(ref['force_first'],-1)
    measured=[mul(v[9:12],1/DT) for v in final]
    stress=numeric_rows(out/'pressure.dat',7)
    particle=numeric_rows(out/'particle_stress.dat',6)
    host_e=numeric_rows(out/'host_energy.dat',1)
    if len(stress)!=1 or len(particle)!=n or len(host_e)!=1:raise ValueError('Unexpected observable time sampling')
    expected_ps=[[0.]*6 for _ in range(n)]
    expected_ps[fx['first']]=mul(ref['virial'],.5);expected_ps[fx['second']]=mul(ref['virial'],.5)
    errors=dict(
        impulse_force_max_abs=compare('Force impulse',measured,forces),
        virial_6_max_abs=compare('Virial',stress[0][1:],ref['virial']),
        particle_stress_max_abs=compare('Per-particle stress',particle,expected_ps),
        particle_sum_max_abs=compare('Sum particle tensors',[sum(v[j] for v in particle) for j in range(6)],stress[0][1:]),
        host_energy_abs=compare('Host total energy',(host_e[0][0]*n,),(ref['energy_total'],)),
        newton_balance_max_abs=compare('Newton balance',[sum(v[j] for v in measured) for j in range(3)],(0.,0.,0.),atol=1e-4,rtol=0.),
    )
    if backend=='CPU':
        native_f=numeric_rows(out/'forces.dat',7)
        if len(native_f)!=n or sorted(int(v[0]) for v in native_f)!=list(range(n)):raise ValueError('CPU force IDs incomplete')
        native_f.sort(key=lambda v:int(v[0]))
        errors['cpu_native_force_max_abs']=compare('Native CPU force',[v[1:4] for v in native_f],forces)
    result=dict(status='PHYSICS_PASS',reference=ref,impulse_forces=measured,
        virial=stress[0][1:],particle_stress=particle,errors=errors,
        host_energy_total=host_e[0][0]*n,optin_environment='ABSENT')
    if backend=='CUDA':
        trace=(out/'native_trace.dat').read_text()
        matches=re.findall(r'(?m)^\s*(\d+)\s+CUDA_energy:\s*([-+0-9.eE]+)\s*$',trace)
        if len(matches)!=1 or int(matches[0][0])!=1:raise ValueError('Native energy not aligned to step1: '+trace)
        native_total=float(matches[0][1])*n
        errors['native_device_energy_abs']=compare('Native device energy',(native_total,),(ref['energy_total'],))
        result.update(native_device_energy_step=1,native_device_energy_total=native_total)
    return result

def self_test():
    out=[]
    for case in CASES:
        if case.negative:continue
        ref=oracle(case)
        if not finite_tree(ref):raise ValueError('Oracle nonfinite')
        rv=add(BACK,mul(RAY,case.r)) if case.pd else mul(RAY,case.r)
        compare('Affine identity',ref['virial'],[-rv[i]*ref['force_first'][j] for i,j in COMP],2e-6,2e-7)
        periodic=mic(add(rv,(BOX,-BOX,BOX)))
        compare('Periodic invariance',(scalar(case,periodic),),(ref['energy_total'],),1e-10,1e-10)
        out.append(dict(case=case.name,**ref))
    return dict(status='PASS',scope='scalar oracle only; executable not run',cases=out)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--binary')
    parser.add_argument('--work',required=True)
    parser.add_argument('--backends',default='CPU,CUDA')
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--filter',default='')
    args=parser.parse_args();work=Path(args.work).resolve();work.mkdir(parents=True,exist_ok=True)
    if args.self_test:
        report=self_test()
        (work/'oracle_self_test.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(status=report['status'],oracle_cases=len(report['cases']))));return 0
    if not args.binary:parser.error('--binary required unless --self-test')
    binary=Path(args.binary).resolve()
    report=dict(test='DNACT_EXCLUDED_PAIR_SCALAR_QUALIFICATION',binary=str(binary),
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),dt=DT,
        tolerances=dict(atol=2e-5,rtol=5e-4),results=[])
    for case in CASES:
        if args.filter and not re.search(args.filter,case.name):continue
        for backend in args.backends.split(','):
            if backend not in ('CPU','CUDA'):raise ValueError(backend)
            if case.negative and backend!='CUDA':continue
            row=dict(case=case.name,backend=backend)
            try:row.update(run_case(binary,case,backend,work/(backend+'_'+case.name)))
            except Exception as exc:row.update(status='FAIL',error=repr(exc))
            report['results'].append(row);print(case.name,backend,row['status'],flush=True)
    report['passed']=sum(r['status'] in ('PHYSICS_PASS','FAIL_CLOSED_PASS') for r in report['results'])
    report['tested']=len(report['results'])
    report['status']='PASS' if report['tested'] and report['passed']==report['tested'] else 'FAIL'
    (work/'excluded_pair_qualification.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('status','passed','tested')}))
    return int(report['status']!='PASS')

if __name__=='__main__':sys.exit(main())
