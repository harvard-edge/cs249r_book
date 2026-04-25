import os
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(8, 3))
blocks = ['Sys Prompt\n(Sink)', 'Past Context', 'Evicted', 'Recent', 'Current\nGen']
values = [1.0, 1.0, 0.0, 1.0, 1.0]
colors = ['#4a90c4', '#4a90c4', '#fdebd0', '#4a90c4', '#3d9e5a']

bars = ax.bar(blocks, values, color=colors, edgecolor='black')
ax.set_title('KV Cache Page Layout with Eviction')
ax.set_yticks([])
ax.set_ylabel('Cache Residency')
fig.tight_layout()
plt.savefig(os.environ.get("VISUAL_OUT_PATH", "out.svg"), format="svg", bbox_inches="tight")
