#!/usr/bin/env python3
"""Reach |L|=0 exactly after first half-kick, despite nonzero initial L."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import angular_fd_oracle as fd
import angular_nve_convergence as nve

def main():
    p=argparse.ArgumentParser(); p.add_argument('--binary',type=Path,required=True); p.add_argument('--backend',choices=('CPU','CUDA'),default='CPU'); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'system.top').write_text('2 1 0 2 0\n-1 K -1 1\n-1 K 0 -1\n')
    (a.out/'springs.par').write_text('2\n0 1 1 s 0 1 0 0.5 1\n')
    (a.out/'init.dat').write_text('t = 0\nb = 20 20 20\nE = 0 0 0\n4 5 6 1 0 0 0 0 1 0 0 0 0 0 -0.00048828125\n5 5 6 0 1 0 0 0 1 0 0 0 0 0 0.00048828125\n')
    text=nve.key(fd.input_text(a.backend,0.,1.,.001953125),'print_energy_every',1)
    (a.out/'input').write_text(text)
    env=os.environ.copy()
    if a.backend=='CUDA': env['OXDNA_EXPERIMENTAL_DNACT_CUDA']='RC2_GLOBAL_ANGULAR_RESEARCH_ONLY'
    proc=subprocess.run([str(a.binary.resolve()),'input'],cwd=a.out,env=env,capture_output=True,text=True,timeout=90)
    (a.out/'stdout.log').write_text(proc.stdout); (a.out/'stderr.log').write_text(proc.stderr)
    result={'backend':a.backend,'exit_code':proc.returncode,'expected_initial_lab_torques':[(0,0,.5),(0,0,-.5)],'initial_body_L':[(0,0,-.00048828125),(0,0,.00048828125)],'dt':.001953125,'expected_halfkick_L':[(0,0,0),(0,0,0)]}
    try:
        state=[[float(v) for v in line.split()] for line in (a.out/'last.dat').read_text().splitlines()[3:] if line.strip()]
        result['final_state']=state; result['finite']=fd.all_finite(state)
        result['status']='PASS' if proc.returncode==0 and result['finite'] else 'FAIL'
    except Exception as e: result.update(status='FAIL',error=str(e))
    # Preserve raw nonfinite tokens in last.dat while emitting strict JSON.
    if 'final_state' in result:
        result['final_state']=[[v if math.isfinite(v) else str(v) for v in row] for row in result['final_state']]
    (a.out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n'); print(json.dumps(result,indent=2,allow_nan=False))
    if result['status']!='PASS': raise SystemExit(1)

if __name__=='__main__': main()
