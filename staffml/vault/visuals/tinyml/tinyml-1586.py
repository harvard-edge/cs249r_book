import os
import matplotlib.pyplot as plt

labels = ['SRAM (Volatile)', 'Flash (Non-Volatile)']
retention = [0, 100]

plt.figure(figsize=(5, 4))
plt.bar(labels, retention, color=['#c87b2a', '#3d9e5a'])
plt.ylabel('Data Retained (%)')
plt.title('Power Loss Scenario')

out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')