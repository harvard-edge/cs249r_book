import os
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(8, 3))
t = np.linspace(0, 400, 4000)
power = np.where(t % 100 < 10, 5, 0.1)
ax.plot(t, power, color='#4a90c4', lw=2)
ax.set_title('VAD Duty Cycle Power Consumption')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Power (mW)')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.savefig(os.environ.get("VISUAL_OUT_PATH", "out.svg"), format="svg", bbox_inches="tight")
