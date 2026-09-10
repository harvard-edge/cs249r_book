import matplotlib.pyplot as plt
import numpy as np
import os

lambda_rate = np.linspace(0.1, 2.4, 50)
mu = 1 / 0.4 # 2.5 req/s
# M/D/1 wait time: rho = lambda/mu. W_q = (rho * service_time) / (2 * (1 - rho))
rho = lambda_rate / mu
wait_time = (rho * 0.4) / (2 * (1 - rho))

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(lambda_rate, wait_time, color='#4a90c4', linewidth=2)
ax.axhline(y=1.6, color='red', linestyle='--', label='1.6s Target Wait')
ax.axvline(x=2.22, color='#3d9e5a', linestyle='--', label='Max Rate (2.22)')
ax.set_xlabel('Arrival Rate (req/s)')
ax.set_ylabel('Queue Wait Time (s)')
ax.set_title('NPU LLM Concurrency Limit')
ax.legend()

out_path = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
fig.savefig(out_path, format='svg', bbox_inches='tight')
plt.close(fig)