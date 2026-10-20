import math
def verify(i,v):
 assert not v['hull_moved'] and v['certified_abreast_mm']==0
 assert math.isclose(v['reach_mm'],1.6*v['age_ms'])
 assert math.isclose(v['certified_ahead_mm'],max(0,950-v['reach_mm']),abs_tol=1e-9)
 if i==4:assert v['expired']
