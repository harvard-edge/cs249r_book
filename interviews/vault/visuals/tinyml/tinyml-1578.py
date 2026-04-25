import os
import matplotlib.pyplot as plt

labels = ['SRAM', 'Flash']
sizes = [0.25, 1.0]
plt.figure(figsize=(6, 4))
plt.bar(labels, sizes, color=['#4a90c4', '#3d9e5a'])
plt.ylabel('Size (MB)')
plt.title('Cortex-M4 Typical Memory')

out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')