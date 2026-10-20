#!/usr/bin/env python3
"""Canonical chapter heading-drift inputs; plot uses tan(theta), matching the PNG."""
import math,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[4]/'mlsysim'))
from mlsysim import Embodied
A=Embodied.Scenario.WarehouseAisle;M=Embodied.MobileManipulator.WarehouseMobileManipulator
D=dict(theta=math.radians(1.0),speed=M.base.max_velocity.to('m/s').magnitude,clearance=A.side_clearance.to('m').magnitude)
D['exhausted']=D['clearance']/math.tan(D['theta']);D['exhausted_s']=D['exhausted']/(D['speed']*math.cos(D['theta']));D['mse']=D['theta']**2
assert abs(D['exhausted']-8.6)<.05
if __name__=='__main__':print(json.dumps(D,sort_keys=True))
