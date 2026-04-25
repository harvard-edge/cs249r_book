import os
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 3))
ax.plot([0, 10, 20, 30, 40], [0, 0.5, 0, 0.5, 0], color='#4a90c4', label='Local Delta')
ax.plot([0, 60], [0, 5], color='#c87b2a', linestyle='--', label='Cloud Full')
ax.set_xlabel('Time (min)')
ax.set_ylabel('Checkpoint Size (GB)')
ax.set_title('Dual-Tier Backup Strategy')
ax.legend()
plt.savefig(os.environ['VISUAL_OUT_PATH'], format='svg', bbox_inches='tight')