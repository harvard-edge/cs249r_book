import os
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4))
# Data: [(start, duration), ...]
m4_tasks = [(0, 20), (20, 20), (40, 20)]
bt_tasks = [(20, 10), (40, 10), (60, 10)]
npu_tasks = [(30, 2), (50, 2), (70, 2)]

for i, (start, duration) in enumerate(m4_tasks):
    ax.barh('M4 Compute', duration, left=start, color='#cfe2f3', edgecolor='#4a90c4')
for i, (start, duration) in enumerate(bt_tasks):
    ax.barh('BT Transfer', duration, left=start, color='#d4edda', edgecolor='#3d9e5a')
for i, (start, duration) in enumerate(npu_tasks):
    ax.barh('NPU Compute', duration, left=start, color='#fdebd0', edgecolor='#c87b2a')

ax.set_xlabel('Time (ms)')
ax.set_title('Pipeline Gantt Chart (3 Inferences)')
plt.grid(axis='x', linestyle='--', alpha=0.7)

out_path = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out_path, format='svg', bbox_inches='tight')
plt.close(fig)