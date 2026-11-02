from pathlib import Path
import sys,re,json,math
R=Path(__file__).resolve().parents[4];sys.path.insert(0,str(R/'mlsysim'))
from mlsysim import *
from mlsysim.core.units import *
from mlsysim.fmt import *
env=dict(globals());env['RM']=Embodied.MobileManipulator.WarehouseMobileManipulator;env['AISLE']=Embodied.Scenario.WarehouseAisle
q=(R/'books/vol4/02_body/02_body.qmd').read_text();cells=re.findall(r'```\{python\}[^\n]*\n(.*?)\n```',q,re.S)
for name in ['CopperResistanceRise', 'ThermalBurst', 'TorqueSpeedEnvelope']:
 cell=next(c for c in cells if 'class '+name+':' in c);exec(compile(cell,name+' chapter cell','exec'),env)
c=env['ThermalBurst'];m=env['TorqueSpeedEnvelope'];d=dict(tau_rated=m.tau_rated.to(newton*meter).magnitude,tau_peak=m.tau_peak.to(newton*meter).magnitude,kt=m.k_t.to(newton*meter/ampere).magnitude,rtt=m.r_tt.to(ohm).magnitude,vbus=m.v_bus.to(volt).magnitude,n0=m.n0.to(rpm).magnitude,i_cont=c.i_cont.to(ampere).magnitude,i_burst=c.i_peak.to(ampere).magnitude,p_burst=c.p_peak.to(watt).magnitude,tau_s=c.tau.to(second).magnitude,ambient_c=c.t_amb.to(degC).magnitude,equilibrium_c=c.t_equil.to(degC).magnitude,asymptote_c=c.t_inf.to(degC).magnitude,insulation_c=c.t_limit.to(degC).magnitude,cold_limit_s=c.t_cold.to(second).magnitude,hot_limit_s=c.t_hot.to(second).magnitude)
print(json.dumps(d))
