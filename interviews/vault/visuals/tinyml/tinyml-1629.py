import os
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 4))
labels = ['Active (0.6%)', 'Sleep (99.4%)']
sizes = [0.6, 99.4]
ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['#ff9999','#66b3ff'])
ax.set_title('Duty Cycle Distribution')
out_path = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
fig.savefig(out_path, format='svg', bbox_inches='tight')
plt.close(fig)