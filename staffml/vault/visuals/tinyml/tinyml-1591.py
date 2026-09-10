import os
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(4,3))
ax.pie([18, 1], labels=['Leakage (18 mAs)', 'Active (1 mAs)'], colors=['#cfe2f3', '#c87b2a'])
ax.set_title('Hourly Energy Usage')
out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')