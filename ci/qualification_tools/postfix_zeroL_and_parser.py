#!/usr/bin/env python3
"""RC3 zero/near-zero half-kick physics and strict numeric-input gates.

Small N=2 cases, no source edits, no build or scheduler calls. Run this against
the new executable on CPU and CUDA. Existing directories are never reused.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import angular_fd_oracle as fd
import angular_nve_convergence as nve


def prepare(out,backend,kb=45.,kt=32.,dt=1e-5):
    out.mkdir(parents=True,exist_ok=False)
    (out/'system.top').write_text('2 1 0 2 0\n-1 K -1 1\n-1 K 0 -1\n')
    (out/'init.dat').write_text(fd.configuration(fd.POSITIONS,fd.FRAMES))
    r=fd.norm(fd.sub(fd.POSITIONS[1],fd.POSITIONS[0]))
    (out/'springs.par').write_text(f'2\n0 1 {r:.17g} s 0 '+ ' '.join(map(str,fd.TARGET))+'\n')
    (out/'input').write_text(fd.input_text(backend,kb,kt,dt))


def execute(binary,backend,out):
    env=os.environ.copy()
    if backend=='CUDA': env['OXDNA_EXPERIMENTAL_DNACT_CUDA']='RC2_GLOBAL_ANGULAR_RESEARCH_ONLY'
    p=subprocess.run([str(binary),'input'],cwd=out,env=env,capture_output=True,text=True,timeout=45)
    (out/'stdout.log').write_text(p.stdout); (out/'stderr.log').write_text(p.stderr)
    log=(out/'log.dat').read_text() if (out/'log.dat').exists() else ''
    return p,log,p.stdout+'\n'+p.stderr+'\n'+log


def read_state(out):
    state=[[float(x) for x in line.split()] for line in (out/'last.dat').read_text().splitlines()[3:] if line.strip()]
    if len(state)!=2 or any(len(x)!=15 for x in state): raise RuntimeError('Final state shape mismatch')
    if not fd.all_finite(state): raise RuntimeError('Final state contains NaN/Inf')
    return state


def zero_case(binary,backend,out,label,delta,zero_initial=False):
    dt=.001953125; half=.00048828125
    prepare(out,backend,0.,1.,dt)
    frames=(((1.,0.,0.),(0.,0.,1.)),((0.,1.,0.),(0.,0.,1.)))
    positions=((4.,5.,6.),(5.,5.,6.))
    Ls=((delta[0],delta[1],-half+delta[2]),(-delta[0],-delta[1],half-delta[2]))
    if zero_initial: Ls=((0.,0.,0.),(0.,0.,0.))
    lines=['t = 0','b = 20 20 20','E = 0 0 0']
    for pos,(a1,a3),L in zip(positions,frames,Ls):
        lines.append(' '.join(f'{x:.17g}' for x in (*pos,*a1,*a3,0.,0.,0.,*L)))
    (out/'init.dat').write_text('\n'.join(lines)+'\n')
    (out/'springs.par').write_text('2\n0 1 1 s 0 1 0 0.5 1\n')
    p,log,full=execute(binary,backend,out)
    if zero_initial:
        if p.returncode==0 or '0 angular momentum in initial configuration' not in full:
            raise RuntimeError('Expected initial-zero-L fail-closed guard not observed')
        return {'label':label,'status':'PASS','behavior':'REJECTED_EXPECTED','exit_code':p.returncode}
    if p.returncode or 'END OF THE SIMULATION, everything went OK!' not in log:
        raise RuntimeError(f'{label}: execution failure: {full[-1500:]}')
    if backend=='CUDA' and 'Running on device 0' not in log: raise RuntimeError('Device not confirmed')
    state=read_state(out)
    atol=3e-11 if backend=='CPU' else 3e-6
    for bead,(row,pos,frame) in enumerate(zip(state,positions,frames)):
        fd.check_close(label+'/position',row[:3],pos,atol,0.)
        # Near-zero half-step angular momentum must produce a rotation close
        # to identity; initial frames and dt are exactly specified.
        fd.check_close(label+'/frame_a1',row[3:6],frame[0],atol,0.)
        fd.check_close(label+'/frame_a3',row[6:9],frame[1],atol,0.)
        fd.check_close(label+'/frame_norms',(fd.norm(row[3:6]),fd.norm(row[6:9]),fd.dot(row[3:6],row[6:9])),(1.,1.,0.),atol,0.)
        sign=1 if bead==0 else -1
        expected_L=(sign*delta[0],sign*delta[1],sign*(half+delta[2]))
        fd.check_close(label+'/final_momentum',row[12:15],expected_L,atol,0.)
    energy=nve.energy_rows(out/'native_energy.dat',backend)
    if not fd.all_finite(energy): raise RuntimeError('Energy contains NaN/Inf')
    return {'label':label,'status':'PASS','exit_code':p.returncode,'dt':dt,'delta_halfkick_L':delta,'final_state':state,'host_energy_rows':energy}


def invalid_case(binary,backend,out,label,key,value):
    prepare(out,backend)
    inp=(out/'input').read_text(); (out/'input').write_text(nve.key(inp,key,value))
    p,log,full=execute(binary,backend,out)
    if p.returncode==0: raise RuntimeError(f'{label}: invalid input accepted with exit code 0')
    if 'END OF THE SIMULATION, everything went OK!' in log: raise RuntimeError(f'{label}: misleading clean completion')
    # Prove rejection belongs to this numeric input, not unavailable hardware
    # or a syntax defect in the shared fixture.
    if not any(x in full.lower() for x in ('non-finite','invalid value','out of range','outside cuda float range','positive and in cuda float range','trailing characters','bending & torsion','temperature should be between')):
        raise RuntimeError(f'{label}: no numeric-rejection diagnostic: {full[-1500:]}')
    return {'label':label,'status':'PASS','behavior':'REJECTED_EXPECTED','key':key,'value':value,'exit_code':p.returncode}


def parser_cases():
    cases=[]
    for key in ('bending_k','torsion_k'):
        for label,value in (('nan','NaN'),('inf','inf'),('signed_inf','-Infinity'),('overflow_double','1e309'),('overflow_float','3.5e38'),('malformed','45junk')):
            cases.append((key+'_'+label,key,value))
    cases.extend((('mixed_negative_kb','bending_k','-1'),('mixed_negative_kt','torsion_k','-1')))
    for key in ('salt_concentration','dh_lambda','dh_strength','debye_huckel_rhigh'):
        for label,value in (('nan','NaN'),('inf','+inf'),('negative','-1'),('zero','0'),('overflow_double','1e309'),('overflow_float','3.5e38'),('malformed','1garbage')):
            cases.append((key+'_'+label,key,value))
    for label,value in (('nan','NaNK'),('inf','infK'),('overflow','1e309K'),('malformed','300junkK'),('zero_kelvin','0K'),('negative_kelvin','-1K')):
        cases.append(('temperature_'+label,'T',value))
    return cases


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--binary',type=Path,required=True);ap.add_argument('--backend',choices=('CPU','CUDA'),default='CPU');ap.add_argument('--out',type=Path,required=True);ap.add_argument('--section',choices=('all','zeroL','parser'),default='all')
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=False);binary=a.binary.resolve()
    result={'backend':a.backend,'binary':str(binary),'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'cases':[],'scope':'N=2 post-fix zero/near-zero angular half-kick plus strict numeric parsing; no NVT or long dynamics claim'}
    failures=[]
    if a.section in ('all','zeroL'):
        for label,delta,initial in [('exact_zero',(0.,0.,0.),False),('near_zero_z',(0.,0.,1e-12),False),('near_zero_oblique',(1e-12,-2e-12,3e-12),False),('initial_zero',(0.,0.,0.),True)]:
            try:result['cases'].append(zero_case(binary,a.backend,a.out/label,label,delta,initial))
            except Exception as e: failures.append(label);result['cases'].append({'label':label,'status':'FAIL','error':str(e)})
    if a.section in ('all','parser'):
        for label,key,value in parser_cases():
            try:result['cases'].append(invalid_case(binary,a.backend,a.out/label,label,key,value))
            except Exception as e:failures.append(label);result['cases'].append({'label':label,'status':'FAIL','error':str(e)})
    result.update(status='FAIL' if failures else 'PASS',failures=failures,passed=sum(c['status']=='PASS' for c in result['cases']),total=len(result['cases']))
    (a.out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))
    if failures:raise SystemExit(1)

if __name__=='__main__':main()
