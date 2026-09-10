import os
import numpy as np
import matplotlib.pyplot as plt

n = np.arange(0, 8)
rho = 0.625
prob = (1 - rho) * (rho**n)
plt.figure(figsize=(6,4))
plt.bar(n, prob, color='#cfe2f3', edgecolor='#4a90c4')
plt.xlabel('Tasks in System (n)')
plt.ylabel('Probability')
plt.title('M/M/1 State Probabilities')
plt.tight_layout()
plt.savefig(os.environ.get('VISUAL_OUT_PATH', 'out.svg'), format='svg', bbox_inches='tight')