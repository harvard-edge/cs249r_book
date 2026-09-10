import os
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5, 3))
intervals = [1, 5, 10, 15, 30]
penalty = [0.5, 2.5, 5.0, 7.5, 15.0]
ax.plot(intervals, penalty, marker='o', color='#3d9e5a')
ax.set_xlabel('Checkpoint Interval (min)')
ax.set_ylabel('Expected Rollback Time (min)')
out_path = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
fig.savefig(out_path, format='svg', bbox_inches='tight')
plt.close(fig)