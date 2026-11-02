from pathlib import Path
import sys,re,json,math
R=Path(__file__).resolve().parents[4];sys.path.insert(0,str(R/'mlsysim'))
from mlsysim import *
from mlsysim.core.units import *
from mlsysim.fmt import *
env=dict(globals());env['RM']=Embodied.MobileManipulator.WarehouseMobileManipulator;env['AISLE']=Embodied.Scenario.WarehouseAisle
q=(R/'books/vol4/04_nervous/04_nervous.qmd').read_text();cells=re.findall(r'```\{python\}[^\n]*\n(.*?)\n```',q,re.S)
for name in ['MultirateStall']:
 cell=next(c for c in cells if 'class '+name+':' in c);exec(compile(cell,name+' chapter cell','exec'),env)
c=env['MultirateStall'];d=dict(renewal_ms=c.t_renew.to(millisecond).magnitude,inference_ms=c.t_infer.to(millisecond).magnitude,lease_ms=c.t_lease.to(millisecond).magnitude,last_admission_ms=c.t_last_adm.to(millisecond).magnitude,expiry_ms=c.t_expiry.to(millisecond).magnitude,brake_onset_ms=c.t_brake.to(millisecond).magnitude)
print(json.dumps(d))
