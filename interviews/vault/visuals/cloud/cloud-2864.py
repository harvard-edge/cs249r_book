import os
import matplotlib.pyplot as plt
import numpy as np
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
cats = ['Total HBM3', 'Allocation']
total = [192, 0]
w = [0, 140]
a = [0, 38.4]
k = [0, 13.6]
ax1.bar(cats, total, label='Total Capacity (192GB)', color='lightgray')
ax1.bar(cats, w, label='Weights (140GB)', color='#cfe2f3')
ax1.bar(cats, a, bottom=w, label='Activations (38.4GB)', color='#fdebd0')
ax1.bar(cats, k, bottom=np.add(w, a), label='KV Cache (13.6GB)', color='#d4edda')
ax1.set_ylabel('Memory (GB)')
ax1.set_title('MI300X HBM3 Allocation')
ax1.legend()
reqs = ['Max Concurrent Requests']
ax2.bar(reqs, [5], color='#4a90c4', width=0.4)
ax2.set_ylabel('Requests')
ax2.set_title('KV Pool Capacity (8192 tokens/req)')
plt.tight_layout()
out = os.environ.get('VISUAL_OUT_PATH', 'plot.svg')
plt.savefig(out, format='svg', bbox_inches='tight')