import os
import matplotlib.pyplot as plt

labels = ['Boot', 'Inference', 'Sleep']
energy = [30, 20, 1.194]
plt.figure(figsize=(5,4))
plt.pie(energy, labels=labels, autopct='%1.1f%%', colors=['#fdebd0', '#c87b2a', '#cfe2f3'])
plt.title('Energy Contribution per Cycle')
plt.tight_layout()
plt.savefig(os.environ.get('VISUAL_OUT_PATH', 'out.svg'), format='svg', bbox_inches='tight')