#!/usr/bin/env python3
"""Canonical capture clock, lease clock, and spatial reservation from Ch12."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[4]/'mlsysim'))
from mlsysim import Embodied
from mlsysim.physics.robotics import calc_c2_stop_suffix,calc_scurve_stop_distance
A=Embodied.Scenario.WarehouseAisle;M=Embodied.MobileManipulator.WarehouseMobileManipulator
ms=lambda q:q.to('ms').magnitude
v=A.v_aisle;j=calc_c2_stop_suffix(M.base.max_velocity,A.a_brake)['peak_jerk'];stop=calc_scurve_stop_distance(v,A.a_brake,j)
delay=A.perception_age+A.t_lease+A.t_tick+A.t_bus+A.t_brake_onset
fixed=A.delta_loc+A.delta_margin+A.eps_track
total=(stop+v*delay+fixed).to('mm');spare=(A.d_clear-total).to('mm');allowance=ms(spare/v)
D=dict(v=v.to('m/s').magnitude,dispatch=ms(A.perception_age),allocation=ms(A.t_dispatch_to_admission),lease=ms(A.t_lease),tick_bus=ms(A.t_tick+A.t_bus),onset_delay=ms(A.t_brake_onset),allowance=allowance,spare=spare.magnitude)
D['admission']=D['dispatch']+D['allocation'];D['expiry']=D['admission']+D['lease'];D['tick_end']=D['expiry']+D['tick_bus'];D['onset']=D['tick_end']+D['onset_delay'];D['latest_admission']=D['dispatch']+D['allowance'];D['latest_onset']=D['latest_admission']+D['lease']+D['tick_bus']+D['onset_delay'];D['reserve_ms']=D['allowance']-D['allocation'];D['reserve_mm']=D['spare']-D['v']*D['allocation']
assert abs(D['onset']-ms(A.t_capture_to_brake_onset))<1e-9 and D['reserve_ms']>0
if __name__=='__main__':print(json.dumps(D,sort_keys=True))
