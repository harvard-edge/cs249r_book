import os
import matplotlib.pyplot as plt
labels = ['Shared RAM Capacity']
weights = [14]
os_mem = [2]
kv_cache = [16]
fig, ax = plt.subplots(figsize=(6,2))
ax.barh(labels, weights, label='Weights', color='#4a90c4')
ax.barh(labels, os_mem, left=weights, label='OS', color='#c87b2a')
ax.barh(labels, kv_cache, left=[16], label='KV Cache', color='#3d9e5a')
ax.legend()
out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')