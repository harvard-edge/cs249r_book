import os
import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots(figsize=(6,3))
ax.bar(['15W Mode', '60W Mode'], [2.25, 3.0], color=['#d4edda', '#fdebd0'], edgecolor=['#3d9e5a', '#c87b2a'])
ax.set_ylabel('Energy per Inference (Joules)')
plt.savefig(os.environ.get('VISUAL_OUT_PATH', 'out.svg'), format='svg', bbox_inches='tight')