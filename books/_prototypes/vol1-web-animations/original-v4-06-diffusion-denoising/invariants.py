def verify(i,v):
 assert v['active_chunk']=='A' and not v['admitted_B']
 assert v['refinement_stage']==min(2,max(0,i-1))
 assert v['filling_chunk']==('B'if i>=4 else'empty')
 assert v['transfer_fraction']==(1 if i>=4 else 0)
