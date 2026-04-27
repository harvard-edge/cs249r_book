import os
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5,4))
ax.bar(['TCM Hit', 'LPDDR5x Spill'], [1, 100], color=['#d4edda', '#fdebd0'], edgecolor=['#3d9e5a', '#c87b2a'])
ax.set_ylabel('Access Latency (ns, Approx)')
ax.set_title('Impact of TCM Miss on NPU')
plt.savefig(os.environ['VISUAL_OUT_PATH'], format='svg', bbox_inches='tight')