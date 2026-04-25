import os
import matplotlib.pyplot as plt

time = [0, 20, 20, 100, 100, 120, 120, 200]
power = [2.5, 2.5, 0, 0, 2.5, 2.5, 0, 0]
plt.figure(figsize=(6, 3))
plt.fill_between(time, power, step='pre', color='#4a90c4', alpha=0.5)
plt.plot(time, power, color='#4a90c4', drawstyle='steps-pre')
plt.ylabel('Power (W)')
plt.xlabel('Time (%)')
plt.title('20% Duty Cycle')

out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')