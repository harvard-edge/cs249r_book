import os
import matplotlib.pyplot as plt

tiers = ['UFS 4.0', 'LPDDR5X', 'NPU SRAM']
bw = [4, 60, 1000]

plt.figure(figsize=(6, 3))
plt.barh(tiers, bw, color=['#c87b2a', '#4a90c4', '#3d9e5a'])
plt.xlabel('Bandwidth (GB/s, log scale)')
plt.xscale('log')
plt.title('Mobile Memory Tier Bandwidths')
plt.savefig(os.environ.get('VISUAL_OUT_PATH', 'out.svg'), format='svg', bbox_inches='tight')