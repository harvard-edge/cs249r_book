from pathlib import Path
import sys,re,json,math
R=Path(__file__).resolve().parents[4];sys.path.insert(0,str(R/'mlsysim'))
from mlsysim import *
from mlsysim.core.units import *
from mlsysim.fmt import *
env=dict(globals());env['RM']=Embodied.MobileManipulator.WarehouseMobileManipulator;env['AISLE']=Embodied.Scenario.WarehouseAisle
q=(R/'books/vol4/13_placement/13_placement.qmd').read_text();cells=re.findall(r'```\{python\}[^\n]*\n(.*?)\n```',q,re.S)
for name in ['PlacementMachine', 'RejectedPlacementThrottle']:
 cell=next(c for c in cells if 'class '+name+':' in c);exec(compile(cell,name+' chapter cell','exec'),env)
c=env['RejectedPlacementThrottle'];d=dict(tau_s=c.tau_th.to(second).magnitude,trip_c=c.t_start,clear_c=c.t_clear,baseline_c=c.t_target,recovery_s=c.t_recov.to(second).magnitude,clock_nom_mhz=c.f_nom.to(megahertz).magnitude,clock_throttled_mhz=c.f_throt.to(megahertz).magnitude,exec_nom_us=c.t_exec.to(microsecond).magnitude,exec_throttled_us=c.t_exec_throt.to(microsecond).magnitude,blocking_us=c.b_plus_i.to(microsecond).magnitude,deadline_us=c.deadline.to(microsecond).magnitude,tick_us=c.tick.to(microsecond).magnitude,cycles=c.n_cycles,base_power_w=c.p_base)
print(json.dumps(d))
