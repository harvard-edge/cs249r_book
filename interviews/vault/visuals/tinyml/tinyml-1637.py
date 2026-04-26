import os
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 3))
t = [0, 2, 2.01, 4, 4.01, 6]
prog = [0, 66, 0, 66, 0, 66]
ax.plot(t, prog, color='#c87b2a', lw=2)
ax.axhline(100, color='r', linestyle='--', label='Task Complete')
ax.set_ylabel('Progress (%)')
ax.set_xlabel('Time (min)')
ax.legend()
out_path = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
fig.savefig(out_path, format='svg', bbox_inches='tight')
plt.close(fig)