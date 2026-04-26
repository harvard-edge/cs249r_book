import os
import matplotlib.pyplot as plt
stages = ['Token', 'Embed', 'NPU Trans.', 'Decode']
times = [1, 2, 10, 3]
plt.bar(stages, times, color=['#cfe2f3', '#cfe2f3', '#c87b2a', '#cfe2f3'])
plt.ylabel('Duration (ms)')
out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')