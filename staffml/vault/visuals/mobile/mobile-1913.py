import os
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6,2))
ax.broken_barh([(0,3)], (0,1), facecolors='#cfe2f3', label='Train')
ax.annotate('WiFi Drop', xy=(3, 0.5), xytext=(2.5, 1.2), arrowprops=dict(facecolor='red', shrink=0.05))
ax.broken_barh([(3,1)], (0,1), facecolors='#c87b2a', label='Serialize')
ax.broken_barh([(6,3)], (0,1), facecolors='#cfe2f3')
ax.set_yticks([])
ax.set_xlabel('Time')
out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')