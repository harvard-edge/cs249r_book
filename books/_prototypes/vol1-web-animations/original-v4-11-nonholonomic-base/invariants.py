import math
def verify(i,v):
 assert v['body_lateral_velocity']==0
 if i in[0,1,2]:assert v['offset_units']==0
 if i in[2,3,4]:assert v['heading_deg']==90
 if i==5:assert v['heading_deg']==0 and v['offset_units']==1.7
 assert v['blocked']==(i==1)
