def verify(i,v):
 if i==0:assert v['sequence']==8 and v['reader']=='AA'
 if 1<=i<5:assert v['sequence']==9 and v['reader']=='AA'
 if i==3:assert v['payload']=='BA' and v['reader']=='AA'
 if i>=5:assert v['sequence']==10
 if i>=6:assert v['reader']=='BB'
 if i>=7:assert v['remaining_ms']==max(0,150-v['time_ms'])
 assert v['expired']==(i>=9) and v['brake_onset']==(i==10) and not v['standstill']
