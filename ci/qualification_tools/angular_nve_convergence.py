#!/usr/bin/env python3
"""Bounded N=2 DNACT NVE convergence check; no long-time qualification."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import angular_fd_oracle as fd

TIMESTEPS=(.002,.001,.0005)
HORIZON=.5
MAX_COARSE_SPAN=2e-3
MAX_FINE_SPAN=2e-4
MAX_FINE_COARSE_RATIO=.5

def key(text,name,value):
    pat=rf"(?m)^\s*{re.escape(name)}\s*=.*$"
    return re.sub(pat,lambda _:f"{name} = {value}",text,count=1) if re.search(pat,text) else text+f"\n{name} = {value}\n"

def energy_rows(path,backend):
    # CUDA_print_energy appends backend_info to the four host observables.
    # Match the entire row, so unexpected tokens/nonfinite values still fail.
    number=r"[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?"
    pattern=rf"\s*({number})\s+({number})\s+({number})\s+({number})"
    if backend=='CUDA': pattern+=rf"(?:\s+CUDA_energy:\s*({number}))?"
    pattern+=r"\s*"
    rows=[]
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'): continue
        match=re.fullmatch(pattern,line)
        if match is None: raise RuntimeError(f'Unexpected energy row in {path}: {line}')
        vals=[float(x) for x in match.groups() if x is not None]
        if not all(math.isfinite(x) for x in vals): raise RuntimeError(f'Nonfinite energy row in {path}')
        rows.append(vals[:4])
    if not rows: raise RuntimeError(f'Empty energy history: {path}')
    return rows

def run_case(binary,backend,out,dt,zero_l=False):
    out.mkdir(parents=True,exist_ok=False)
    (out/'system.top').write_text('2 1 0 2 0\n-1 K -1 1\n-1 K 0 -1\n')
    conf=fd.configuration(fd.POSITIONS,fd.FRAMES)
    if zero_l:
        lines=conf.splitlines()
        for j in (3,4): lines[j]=' '.join(lines[j].split()[:12]+['0','0','0'])
        conf='\n'.join(lines)+'\n'
    (out/'init.dat').write_text(conf)
    r=fd.norm(fd.sub(fd.POSITIONS[1],fd.POSITIONS[0]))
    (out/'springs.par').write_text(f'2\n0 1 {r:.17g} s 0 '+ ' '.join(map(str,fd.TARGET))+'\n')
    text=fd.input_text(backend,0. if zero_l else 45.,0. if zero_l else 32.,dt)
    steps=1 if zero_l else round(HORIZON/dt)
    for k,v in [('steps',steps),('print_energy_every',1),('print_conf_interval',steps+1),('seed',20261026)]: text=key(text,k,v)
    (out/'input').write_text(text)
    env=os.environ.copy()
    if backend=='CUDA': env['OXDNA_EXPERIMENTAL_DNACT_CUDA']='RC2_GLOBAL_ANGULAR_RESEARCH_ONLY'
    p=subprocess.run([str(binary),'input'],cwd=out,env=env,capture_output=True,text=True,timeout=120)
    (out/'stdout.log').write_text(p.stdout); (out/'stderr.log').write_text(p.stderr)
    log=(out/'log.dat').read_text() if (out/'log.dat').exists() else ''
    if zero_l and p.returncode!=0 and '0 angular momentum in initial configuration' in log:
        return {'status':'PASS','behavior':'REJECTED_EXPECTED','backend':backend,'test':'zero_L_zero_torque_initial_guard','exit_code':p.returncode}
    if p.returncode or 'END OF THE SIMULATION, everything went OK!' not in log: raise RuntimeError(f'Failed {backend} dt={dt}: {p.stderr[-1200:]} {log[-1200:]}')
    if backend=='CUDA' and 'Running on device 0' not in log: raise RuntimeError('CUDA not confirmed')
    state=[[float(x) for x in line.split()] for line in (out/'last.dat').read_text().splitlines()[3:] if line.strip()]
    if len(state)!=2 or any(len(x)!=15 for x in state) or not fd.all_finite(state): raise RuntimeError('Final state nonfinite or malformed')
    if zero_l:
        for actual,expected in zip(state,fd.POSITIONS): fd.check_close('stationary zero-L position',actual[:3],expected,2e-6,1e-7)
        return {'status':'PASS','backend':backend,'test':'zero_L_zero_torque_one_step','steps':1}
    rows=energy_rows(out/'native_energy.dat',backend)
    if len(rows)!=steps+1: raise RuntimeError(f'Incomplete energy history: {len(rows)}, expected {steps+1}')
    e=[row[3] for row in rows]; denom=max(1.,abs(e[0])); span=(max(e)-min(e))/denom
    result={'backend':backend,'dt':dt,'steps':steps,'duration':steps*dt,'samples':len(rows),
            'energy_initial_per_particle':e[0],'energy_final_per_particle':e[-1],
            'relative_total_energy_span':span,'relative_endpoint_drift':abs(e[-1]-e[0])/denom}
    if backend=='CUDA':
        trace=(out/'native_backend_trace.dat').read_text()
        matches=re.findall(r'(?m)^\s*(\d+)\s+CUDA_energy:\s*([-+0-9.eE]+)\s*$',trace)
        if len(matches)!=steps or [int(x[0]) for x in matches]!=list(range(1,steps+1)): raise RuntimeError('Incomplete native CUDA trace')
        native=[float(x[1]) for x in matches]
        if not all(math.isfinite(x) for x in native): raise RuntimeError('Nonfinite native CUDA trace')
        err=fd.check_close('time-aligned native CUDA potential vs host potential',native,[row[1] for row in rows[1:]],2e-4,2e-5)
        native_total=[u+row[2] for u,row in zip(native,rows[1:])]
        result.update(native_cuda_host_potential_max_abs_error=err,native_cuda_total_energy_span=(max(native_total)-min(native_total))/denom)
    return result

def main():
    p=argparse.ArgumentParser(); p.add_argument('--binary',type=Path,required=True); p.add_argument('--backend',choices=('CPU','CUDA'),default='CPU'); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True); binary=a.binary.resolve()
    result={'scope':'N=2 angular NVE microtest only; duration 0.5; no claim for KDPG, NVT, long dynamics, sorting, pressure coupling or other platforms',
            'predeclared_criteria':{'dt':TIMESTEPS,'duration':HORIZON,'coarse_span_max':MAX_COARSE_SPAN,'fine_span_max':MAX_FINE_SPAN,'fine_coarse_span_ratio_max':MAX_FINE_COARSE_RATIO},
            'binary':str(binary),'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest()}
    # Execute the stationary test separately, retaining a failure without
    # suppressing the physically independent NVE convergence measurements.
    try: result['zero_L']=run_case(binary,a.backend,a.out/'zero_L',1e-5,zero_l=True)
    except Exception as e: result['zero_L']={'status':'FAIL','error':str(e)}
    try:
        cases=[run_case(binary,a.backend,a.out/f'dt_{dt}',dt) for dt in TIMESTEPS]; result['cases']=cases
        coarse=cases[0]['relative_total_energy_span']; fine=cases[-1]['relative_total_energy_span']; ratio=fine/max(coarse,1e-30); result['fine_coarse_span_ratio']=ratio
        if coarse>MAX_COARSE_SPAN or fine>MAX_FINE_SPAN: raise RuntimeError('Energy conservation criterion failed')
        if coarse>2e-6 and ratio>MAX_FINE_COARSE_RATIO: raise RuntimeError('Timestep convergence criterion failed')
        result['NVE_status']='PASS'
    except Exception as e: result.update(NVE_status='FAIL',error=str(e))
    result['status']='PASS' if result['NVE_status']=='PASS' and result['zero_L']['status']=='PASS' else 'FAIL'
    (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
    if result['status']!='PASS': raise SystemExit(1)

if __name__=='__main__': main()
