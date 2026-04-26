import os
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5,4))
ax.bar(['16-token', '256-token'], [0.05, 0.45], color=['#cfe2f3', '#fdebd0'], edgecolor=['#4a90c4', '#c87b2a'])
ax.set_ylabel('Internal Fragmentation Ratio')
ax.set_title('PagedAttention KV Cache Fragmentation')
plt.savefig(os.environ['VISUAL_OUT_PATH'], format='svg', bbox_inches='tight')