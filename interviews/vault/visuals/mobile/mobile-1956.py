import os
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 3))
ax.barh(['Turn 1', 'Turn 2', 'Turn 3'], [3000, 3000, 3000], color='#cfe2f3', label='System Prompt KV')
ax.barh(['Turn 1', 'Turn 2', 'Turn 3'], [10, 10, 10], left=[3000, 3000, 3000], color='#d4edda', label='Generated KV')
ax.set_xlabel('Tokens')
ax.legend()
out_path = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
fig.savefig(out_path, format='svg', bbox_inches='tight')
plt.close(fig)