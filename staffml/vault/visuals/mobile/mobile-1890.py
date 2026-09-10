import os
import numpy as np
import matplotlib.pyplot as plt

n = np.arange(0, 10)
rho = 0.75
prob = (1 - rho) * (rho**n)
plt.figure(figsize=(6,4))
plt.bar(n[:5], prob[:5], color='#cfe2f3', label='NPU')
plt.bar(n[5:], prob[5:], color='#fdebd0', label='CPU Fallback')
plt.xlabel('Tasks in System')
plt.ylabel('Probability')
plt.legend()
plt.tight_layout()
plt.savefig(os.environ.get('VISUAL_OUT_PATH', 'out.svg'), format='svg', bbox_inches='tight')