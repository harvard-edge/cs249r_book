import os
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 4))
labels = ['L1 Cache (Est)', 'L2 Cache (Est)', 'LPDDR5']
bw = [2000, 800, 204.8]
ax.bar(labels, bw, color=['#cfe2f3', '#d4edda', '#fdebd0'], edgecolor='#4a90c4')
ax.set_ylabel('Bandwidth (GB/s)')
ax.set_title('AGX Orin Memory Tier Bandwidth')
for i, v in enumerate(bw):
    ax.text(i, v + 50, str(v), ha='center')
plt.savefig(os.environ['VISUAL_OUT_PATH'], format='svg', bbox_inches='tight')