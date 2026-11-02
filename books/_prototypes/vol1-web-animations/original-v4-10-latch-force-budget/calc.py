#!/usr/bin/env python3
"""Use the exact contact profile and operands from the owning chapter/MLSysIM."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[4]/'mlsysim'))
from mlsysim import Embodied
from mlsysim.physics.robotics import calc_latch_contact_stopping_profile
A=Embodied.Scenario.WarehouseAisle;S=calc_latch_contact_stopping_profile(scenario=A)
D=dict(k=A.k_latch.to('N/m').magnitude,v=A.v_latch_approach.to('m/s').magnitude,trip=A.f_latch_tripwire.to('N').magnitude,limit=A.f_latch_limit.to('N').magnitude,response=A.t_contact_response.to('ms').magnitude,onset_force=S['f_at_brake_onset'].to('N').magnitude,a=S['a_arm_needed'].to('m/s**2').magnitude,stop_ms=S['t_arm_stop'].to('ms').magnitude)
D['trip_ms']=1000*D['trip']/(D['k']*D['v']);D['onset_ms']=D['trip_ms']+D['response'];D['end_ms']=D['onset_ms']+D['stop_ms'];D['unbraked_limit_ms']=1000*D['limit']/(D['k']*D['v'])
assert D['onset_force']==39 and abs(D['k']*(D['v']*D['stop_ms']/1000-.5*D['a']*(D['stop_ms']/1000)**2)+D['onset_force']-D['limit'])<1e-9
if __name__=='__main__':print(json.dumps(D,sort_keys=True))
