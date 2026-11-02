"""Export canonical chapter inputs; requires the project's MLSysIM environment."""
from pathlib import Path
import re,json,math
root=Path(__file__).resolve().parents[4]
import sys
sys.path.insert(0,str(root/'mlsysim'))
from mlsysim import *
from mlsysim.core.units import *
from mlsysim.fmt import *
env=dict(globals())
q=(root/'books/vol4/09_memory/09_memory.qmd').read_text()
cell=next(c for c in re.findall(r'```\{python\}[^\n]*\n(.*?)\n```',q,re.S) if 'class LatchTransformStaleness:' in c)
exec(compile(cell,'LatchTransformStaleness chapter cell','exec'),env)
c=env['LatchTransformStaleness']
d=dict(frame_ms=c.dt.to(millisecond).magnitude,initial_mm=c.e0.to(millimeter).magnitude,drift_mm_s=c.v_drift.to(millimeter/second).magnitude,lever_mm=c.lever.to(millimeter).magnitude,omega_rad_s=c.omega.to(radian/second).magnitude,tolerance_mm=c.e_max.to(millimeter).magnitude)
print(json.dumps(d))
