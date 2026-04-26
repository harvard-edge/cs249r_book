import matplotlib.pyplot as plt
import os

labels = ['Active Phase', 'Sleep Phase']
# Values in mA*seconds (mC)
energy = [15 * 10, 0.002 * 86390]
colors = ['#c87b2a', '#cfe2f3']

fig, ax = plt.subplots(figsize=(4, 4))
ax.pie(energy, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
ax.set_title('Daily Energy Split (mAh)')

out_path = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
fig.savefig(out_path, format='svg', bbox_inches='tight')
plt.close(fig)