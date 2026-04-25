import os
import matplotlib.pyplot as plt

bw_types = ['Orin Available', 'Required (20 tok/s)']
values = [204.8, 280]

plt.figure(figsize=(6, 3))
plt.barh(bw_types, values, color=['#3d9e5a', '#c87b2a'])
plt.xlabel('Bandwidth (GB/s)')
plt.axvline(204.8, color='black', linestyle='--')
plt.title('Bandwidth Requirement vs Hardware Limit')
plt.savefig(os.environ.get('VISUAL_OUT_PATH', 'out.svg'), format='svg', bbox_inches='tight')