import math
def verify(i,v):
 assert math.isclose(v['temperature_c'],53+42*math.exp(-v['cooling_s']/8))
 assert v['clock_restored']==(i>=5)
 assert v['cycles']==300000 and v['deadline_us']==400
 assert v['response_us']==(340 if i>=5 else 590)
 assert v['execution_us']==(250 if i>=5 else 500)
