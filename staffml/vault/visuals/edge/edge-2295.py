import os
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5,3))
ax.barh(['Seq A', 'Seq B', 'Seq C'], [64, 32, 48], color='#cfe2f3', edgecolor='#4a90c4')
ax.set_xlabel('Allocated Pages')
plt.title('Paged KV Cache Allocation')
out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')