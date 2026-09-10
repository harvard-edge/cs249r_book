import os
import matplotlib.pyplot as plt

time = [0, 50, 50, 1000, 1000, 1050, 1050, 2000]
state = [1, 1, 0, 0, 1, 1, 0, 0]
plt.figure(figsize=(6, 2))
plt.plot(time, state, color='#c87b2a', drawstyle='steps-pre')
plt.yticks([0, 1], ['Sleep', 'Active'])
plt.xlabel('Time (ms)')
plt.title('Duty Cycle Timeline')

out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')