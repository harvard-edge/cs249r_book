import os
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 3))
t = [0, 1, 6, 10]
occ = [0, 0, 100, 0]
ax.plot(t, occ, color='#3d9e5a', lw=2)
ax.axvline(1, color='r', linestyle='--', alpha=0.5)
ax.axvline(6, color='r', linestyle='--', alpha=0.5)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Frames in Queue')
ax.set_title('Queue Occupancy')
out_path = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
fig.savefig(out_path, format='svg', bbox_inches='tight')
plt.close(fig)