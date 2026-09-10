import os
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(['FP16 KV'], [1.0], label='Weights', color='#cfe2f3')
ax.bar(['FP16 KV'], [2.0], bottom=[1.0], label='FP16 KV Cache', color='#4a90c4')
ax.bar(['INT8 KV'], [1.0], color='#cfe2f3')
ax.bar(['INT8 KV'], [1.0], bottom=[1.0], label='INT8 KV Cache', color='#3d9e5a')
ax.axhline(3.0, color='r', linestyle='--', label='3GB OS Limit')
ax.legend()
out_path = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
fig.savefig(out_path, format='svg', bbox_inches='tight')
plt.close(fig)