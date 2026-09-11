import os
import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots(figsize=(6,3))
stages = np.arange(4)
for s in stages:
    ax.barh(s, 10, left=s, color='#cfe2f3', edgecolor='#4a90c4')
    ax.barh(s, s, left=0, color='#fdebd0', edgecolor='#c87b2a', hatch='//')
ax.set_ylabel('Pipeline Stage')
ax.set_xlabel('Time Steps')
ax.set_yticks(stages)
plt.savefig(os.environ.get('VISUAL_OUT_PATH', 'out.svg'), format='svg', bbox_inches='tight')