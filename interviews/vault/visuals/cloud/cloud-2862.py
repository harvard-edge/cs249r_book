import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

time = np.arange(0, 60)
state = []
for t in time:
    if 0 <= t < 10: state.append(1)
    elif 10 <= t < 15: state.append(2)
    elif 15 <= t < 25: state.append(1)
    elif 25 <= t < 30: state.append(2)
    elif 30 <= t < 35: state.append(1)
    elif t == 35: state.append(0)
    elif 35 < t <= 45: state.append(3)
    else: state.append(2)

colors = {0: 'red', 1: '#4a90c4', 2: '#3d9e5a', 3: '#c87b2a'}
fig, ax = plt.subplots(figsize=(10, 2))
for t, s in enumerate(state):
    ax.barh(0, 1, left=t, color=colors[s], edgecolor='none')
ax.set_yticks([])
ax.set_xlabel('Time (Minutes)')
ax.set_title('Synchronous Checkpointing Overload (C=10, M=11.25)')
legend_elements = [Patch(facecolor='#3d9e5a', label='Compute'), Patch(facecolor='#4a90c4', label='Checkpoint'), Patch(facecolor='red', label='Crash'), Patch(facecolor='#c87b2a', label='Recovery')]
ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
plt.tight_layout()
plt.savefig(os.environ.get('VISUAL_OUT_PATH', 'out.svg'), format='svg', bbox_inches='tight')