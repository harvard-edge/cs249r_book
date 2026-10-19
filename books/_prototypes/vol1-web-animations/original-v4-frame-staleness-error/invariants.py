import math
def verify(i,v):
 fraction=min(i,4)/4
 assert math.isclose(v['age_ms'],1000/60*fraction)
 assert math.isclose(v['translation_mm'],12*v['age_ms']/1000)
 assert math.isclose(v['rotation_bound_mm'],150*.30*v['age_ms']/1000)
 assert math.isclose(v['error_bound_mm'],.1+v['translation_mm']+v['rotation_bound_mm'])
 assert v['expired']==(i==5)
 if i==4:assert v['error_bound_mm']>v['tolerance_mm']
