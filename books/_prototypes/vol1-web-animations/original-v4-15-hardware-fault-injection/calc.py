#!/usr/bin/env python3
"""Export canonical chapter LEGO arithmetic; no browser hardware registry."""
import re,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT/'mlsysim'))
source=ROOT/'books/vol4/15_verification/15_verification.qmd'
namespace={}
for cell in re.findall(r'```\{python\}[^\n]*\n(.*?)\n```',source.read_text(),re.S):
 if any('class '+name+':' in cell for name in ['RackEndVerification']):exec(compile(cell,str(source),'exec'),namespace)
R=namespace['RackEndVerification'];ms=lambda v:float(v.to('ms').magnitude);mm=lambda v:float(v.to('mm').magnitude)
from mlsysim.physics.robotics import calc_scurve_stop_time
stop=calc_scurve_stop_time(R.v_aisle,R.a_brake,R.j_brake).to('s').magnitude
data=dict(v=R.v_aisle.to('m/s').magnitude,a=R.a_brake.to('m/s**2').magnitude,j=R.j_brake.to('m/s**3').magnitude,stop_s=stop,lease_ms=ms(R.t_lease),command_ms=ms(R.t_lease+R.t_tick),frame_ms=ms(R.t_lease+R.t_tick+R.t_bus),onset_ms=ms(R.onset_bound),late_ms=ms(R.onset_late),rest_late_ms=ms(R.onset_late)+stop*1000,onset_margin_ms=ms(R.rho_onset),travel_margin_mm=mm(R.rho_dist),travel_late_mm=mm(R.travel_late),travel_bound_mm=mm(R.f1_aisle_stop_bound))
if __name__=='__main__':
 out=Path(__file__).with_name('data.json');out.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data,indent=2))
