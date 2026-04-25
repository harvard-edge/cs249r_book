import os
import matplotlib.pyplot as plt

stages = [1, 2, 3, 4]
start_times = [0, 1, 2, 3]
durations = [4, 4, 4, 4]

fig, ax = plt.subplots(figsize=(6, 4))
for i in range(4):
    ax.barh(stages[i], durations[i], left=start_times[i], color='#3d9e5a')
    ax.barh(stages[i], start_times[i], left=0, color='#fdebd0', hatch='//')
ax.set_yticks(stages)
ax.set_yticklabels(['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4'])
ax.set_xlabel('Time (Microbatches)')
ax.set_title('GPipe Schedule')

out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')