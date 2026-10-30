#!/usr/bin/env python3
"""Semantic release gate: process exit zero alone never establishes a pass."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--binary',type=Path,required=True)
p.add_argument('--backend',choices=('CPU','CUDA'),default='CPU')
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();root=a.output.resolve();root.mkdir(parents=True,exist_ok=False)
binary=a.binary.resolve();tools=Path(__file__).resolve().parent/'qualification_tools'
assert binary.is_file() and os.access(binary,os.X_OK)
tests=[
 ('angular_fd','angular_fd_oracle.py',['--backend',a.backend,'--out'], 'result.json','status','PASS'),
 ('nve','angular_nve_convergence.py',['--backend',a.backend,'--out'],'result.json','status','PASS'),
 ('halfkick','zero_halfkick_reproducer.py',['--backend',a.backend,'--out'],'result.json','status','PASS'),
 ('ev','run_ev_impulse.py',['--backends',a.backend,'--work'],'ev_impulse_report.json','scientific_all_pass',True),
 ('inputs','run_input_safety.py',['--work'],'input_safety_report.json','all_rejected',True),
 ('excluded_pp_dh','probe_excluded_pp_dh.py',['--backends',a.backend,'--work'],'excluded_pp_dh_report.json','status','PASS'),
 ('postfix','postfix_zeroL_and_parser.py',['--backend',a.backend,'--out'],'result.json','status','PASS'),
 ('excluded_pairs','qualify_excluded_pairs.py',['--backends',a.backend,'--work'],'excluded_pair_qualification.json','status','PASS'),
 ('global_coexistence','global_angular_es_coexistence.py',['--backend',a.backend,'--out'],'result.json','status','PASS')]
results=[]
for name,script,options,report_name,key,expected in tests:
 out=root/name;cmd=[sys.executable,'-B',str(tools/script),'--binary',str(binary),*options,str(out)]
 item={'test':name,'command':cmd,'passed':False}
 try:
  with (root/(name+'.log')).open('w') as log:run=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=180)
  data=json.loads((out/report_name).read_text())
  item.update(exit_code=run.returncode,semantic_value=data.get(key),passed=run.returncode==0 and data.get(key)==expected)
 except (OSError,ValueError,subprocess.TimeoutExpired) as exc:item['error']=str(exc)
 results.append(item);print(name,'PASS' if item['passed'] else 'FAIL',flush=True)
report={'backend':a.backend,'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'tests':results,'all_passed':all(x['passed'] for x in results),'public_release_authorized':False}
(root/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
sys.exit(0 if report['all_passed'] else 1)
