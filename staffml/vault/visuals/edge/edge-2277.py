import os
import matplotlib.pyplot as plt
import numpy as np
tiers = ['L1 Cache', 'L2 Cache', 'LPDDR5 Main']
bw = [2000, 800, 204.8]
plt.barh(tiers, bw, color=['#fdebd0', '#d4edda', '#cfe2f3'])
plt.xlabel('Theoretical Bandwidth (GB/s)')
out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')