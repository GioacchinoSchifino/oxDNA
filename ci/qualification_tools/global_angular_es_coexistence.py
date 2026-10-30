#!/usr/bin/env python3
"""Nonzero global angular plus PD/PP additivity; explicit native step-1 energy."""
import argparse,importlib.util,json,math,os,re,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--binary',required=True);p.add_argument('--backend',choices=('CPU','CUDA'),default='CPU');p.add_argument('--out',type=Path,required=True);a=p.parse_args();root=a.out.resolve();root.mkdir(parents=True,exist_ok=False)
os.environ['OXDNA_BIN']=str(Path(a.binary).resolve());os.environ.pop('OXDNA_EXPERIMENTAL_DNACT_CUDA',None);os.environ.pop('OXDNA_EXPERIMENTAL_SORT_CUDA',None)
harness=Path(__file__).resolve().parents[2]/'test/DNANM/protein_electrostatics/test_integrated_cpu_cuda_parity.py'
spec=importlib.util.spec_from_file_location('coexist_harness',harness);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
m.POSITIONS=((5.,5.,5.),(4.660,5.341,6.020),(5.260,5.540,6.290),(5.720,5.900,6.380))
def configuration():
 rows=[]
 for i,pos in enumerate(m.POSITIONS):
  alpha,beta=((0.,0.),(.14,.18),(-.23,-.15),(.29,.27))[i];ca,sa,cb,sb=math.cos(alpha),math.sin(alpha),math.cos(beta),math.sin(beta)
  vals=(*pos,ca*cb,sa*cb,-sb,ca*sb,sa*sb,cb,0.,0.,0.,*m.L_INIT);rows.append(' '.join(f'{v:.16g}' for v in vals))
 return 't = 0\nb = 20 20 20\nE = 0 0 0\n'+'\n'.join(rows)+'\n'
m.configuration=configuration;base=m.input_file;angular=False
extra='''
data_output_3 = {
 name = host_u.dat
 print_every = 1
 col_1 = { type = step }
 col_2 = { type = potential_energy
 precision = 17 }
}
data_output_4 = {
 name = native_trace.dat
 print_every = 1
 col_1 = { type = step }
 col_2 = { type = backend_info }
}
'''
def input_file(backend,mode,pd,pp):
 text=base(backend,mode,pd,pp)
 return text.replace('bending_k = 0.0','bending_k = '+('45.0' if angular else '0.0')).replace('torsion_k = 0.0','torsion_k = '+('32.0' if angular else '0.0'))+extra
m.input_file=input_file
def close(x,y,atol=5e-4,rtol=1e-3):
 assert len(x)==len(y)>0
 for u,v in zip(x,y):
  if isinstance(u,(tuple,list)):close(u,v,atol,rtol)
  else:assert math.isfinite(u) and math.isfinite(v) and abs(u-v)<=atol+rtol*abs(v),(u,v)
def plus(x,y):return tuple(plus(a,b) if isinstance(a,(tuple,list)) else a+b for a,b in zip(x,y))
states={};energies={};native=[]
for ang,es in ((0,0),(1,0),(0,1),(1,1)):
 angular=bool(ang);label=f'G{ang}_E{es}'
 old=m.input_file
 if a.backend=='CUDA':
  def with_native(*args):return old(*args)+'\nCUDA_print_energy = true\n'
  m.input_file=with_native
 z=m.run(root,backend=a.backend,mode='DNACT',label=label,charges=(1.,0.,1.),pd=bool(es),pp=bool(es));m.input_file=old
 assert len(z['state'])==m.N and all(len(q)==15 for q in z['state'])
 assert len(z['stress'])==6 and len(z['particle_stress'])==m.N and all(len(q)==6 for q in z['particle_stress'])
 folder=root/f'{a.backend}_DNACT_{label}';rows=[list(map(float,line.split())) for line in (folder/'host_u.dat').read_text().splitlines() if line.strip() and not line.lstrip().startswith('#')]
 assert len(rows)==2 and [int(row[0]) for row in rows]==[0,1] and all(len(row)==2 and all(map(math.isfinite,row)) for row in rows)
 energies[(ang,es)]=m.N*rows[0][1]
 if a.backend=='CUDA':
  matches=re.findall(r'(?m)^\s*(\d+)\s+CUDA_energy:\s*([-+0-9.eE]+)\s*$',(folder/'native_trace.dat').read_text());assert len(matches)==1 and int(matches[0][0])==1
  value=float(matches[0][1])*m.N;reference=rows[1][1]*m.N;assert math.isfinite(value) and abs(value-reference)<=2e-4+2e-5*abs(reference)
  native.append({'angular':ang,'electrostatics':es,'step':1,'native_energy':value,'host_same_step_energy':reference,'absolute_error':abs(value-reference)})
 states[(ang,es)]=z
responses=[m.response(states[(0,0)],states[key]) for key in ((1,0),(0,1),(1,1))]
for field in ('forces','torques','stress','particle_stress'):close(responses[2][field],plus(responses[0][field],responses[1][field]))
for response in responses:close(m.particle_sum(response['particle_stress']),response['stress'])
assert max(abs(v) for q in responses[0]['forces'] for v in q)>1e-3
assert max(abs(v) for q in responses[1]['forces'] for v in q)>1e-5
err=abs(energies[(1,1)]-energies[(1,0)]-energies[(0,1)]+energies[(0,0)]);assert err<1e-9
report={'status':'PASS','backend':a.backend,'global_stiffness':[45,32],'environment_optin_used':False,'host_initial_energy_additivity_abs_error':err,'native_energy_checks':native,'scope':'One-step nonzero angular + PD/PP additivity, force/torque/stress/particle stress; not long dynamics'}
(root/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2))
