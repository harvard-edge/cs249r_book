import os
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 2))
plt.barh(['Hailo-8'], [60], color='#4a90c4')
plt.xlim(0, 100)
plt.axvline(100, color='red', linestyle='--')
plt.xlabel('Utilization (%)')

out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')