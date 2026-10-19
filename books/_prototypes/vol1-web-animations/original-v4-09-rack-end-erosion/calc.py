from pathlib import Path
import sys,re,json,math
R=Path(__file__).resolve().parents[4];sys.path.insert(0,str(R/'mlsysim'))
from mlsysim import *
from mlsysim.core.units import *
from mlsysim.fmt import *
env=dict(globals());env['RM']=Embodied.MobileManipulator.WarehouseMobileManipulator;env['AISLE']=Embodied.Scenario.WarehouseAisle
q=(R/'books/vol4/09_memory/09_memory.qmd').read_text();cells=re.findall(r'```\{python\}[^\n]*\n(.*?)\n```',q,re.S)
for name in ['RackEndBelief']:
 cell=next(c for c in cells if 'class '+name+':' in c);exec(compile(cell,name+' chapter cell','exec'),env)
c=env['RackEndBelief'];d=dict(clear_mm=c.d_clear.to(millimeter).magnitude,localization_mm=c.delta_loc.to(millimeter).magnitude,margin_mm=c.delta_margin.to(millimeter).magnitude,speed_mm_s=c.v_h.to(millimeter/second).magnitude,dispatch_ms=c.t_age.to(millisecond).magnitude,expiry_ms=c.t_exp_ahead.to(millisecond).magnitude)
print(json.dumps(d))
