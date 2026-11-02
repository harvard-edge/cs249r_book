import math
def verify(i,v):
 assert v['requested_torque_nm']<4.05
 assert v['voltage_feasible']==(i!=2)
 if i in[5,7]:assert math.isclose(v['temperature_c'],155,abs_tol=1e-9)
 assert v['hot_start']==(i>=6)
 if i==2:assert v['heating_w']==0
