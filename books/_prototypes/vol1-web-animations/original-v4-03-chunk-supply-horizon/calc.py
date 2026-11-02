#!/usr/bin/env python3
"""Canonical direct-replacement chunk schedule and tracked-evidence horizon."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[4]/'mlsysim'))
from mlsysim import Embodied
from mlsysim.physics.robotics import calc_tracked_evidence_horizon
A=Embodied.Scenario.WarehouseAisle;M=Embodied.MobileManipulator.WarehouseMobileManipulator
ms=lambda q:q.to('ms').magnitude
S=M.chunk_schedule
D=dict(period=ms(S['renewal_period']),latency=ms(M.chunk_inference_latency),step=ms(M.chunk_step),span=ms(S['chunk_span']),replacement=ms(S['open_loop_span']),horizon=ms(calc_tracked_evidence_horizon(A.grasp_tolerance,A.grasp_initial_error,A.a_conveyor_slip)),slip=A.a_conveyor_slip.to('m/s**2').magnitude,initial=A.grasp_initial_error.to('mm').magnitude,tolerance=A.grasp_tolerance.to('mm').magnitude,min_supply=S['min_supply_setpoints'])
assert D['replacement']<D['horizon']<D['span'] and D['min_supply']==4
if __name__=='__main__':print(json.dumps(D,sort_keys=True))
