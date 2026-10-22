#!/usr/bin/env python3
"""Export canonical chapter LEGO arithmetic; no browser hardware registry."""
import re,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT/'mlsysim'))
source=ROOT/'books/vol4/10_intent/10_intent.qmd'
namespace={}
for cell in re.findall(r'```\{python\}[^\n]*\n(.*?)\n```',source.read_text(),re.S):
 if any('class '+name+':' in cell for name in ['IntentLeaseArchetypes']):exec(compile(cell,str(source),'exec'),namespace)
R=namespace['IntentLeaseArchetypes'];A=R.AISLE
data=dict(untracked_ms=R.tau_ev_untracked.magnitude,tracked_ms=R.tau_ev_tracked.magnitude,stall_ms=R.delta_t_stall_intent.magnitude,inference_ms=R.RM.intent_inference_latency.to('ms').magnitude,e0_mm=A.grasp_initial_error.to('mm').magnitude,belt_mm_ms=A.v_conveyor.to('mm/ms').magnitude,slip_mm_ms2=A.a_conveyor_slip.to('mm/ms**2').magnitude,tolerance_mm=A.grasp_tolerance.to('mm').magnitude,stop_after_ms=R.t_arm_onset_after_expiry.magnitude)
if __name__=='__main__':
 out=Path(__file__).with_name('data.json');out.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data,indent=2))
