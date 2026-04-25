import os
import matplotlib.pyplot as plt
import numpy as np

T = np.linspace(50, 1000, 100)
C = 50
lam = 1/720
cost_waste = C / T
cost_fail = lam * T / 2
cost_total = cost_waste + cost_fail

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(T, cost_waste * 100, label='Checkpoint Overhead', color='#4a90c4', ls='--')
ax.plot(T, cost_fail * 100, label='Failure Recompute Time', color='#c87b2a', ls='-.')
ax.plot(T, cost_total * 100, label='Total Expected Waste', color='#3d9e5a', lw=2)
ax.axvline(np.sqrt(2*C/lam), color='r', linestyle=':', label='Optimal Interval')
ax.set_title('Expected Wasted Time vs Checkpoint Interval')
ax.set_xlabel('Checkpoint Interval (seconds)')
ax.set_ylabel('% Wasted Time')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.savefig(os.environ.get("VISUAL_OUT_PATH", "out.svg"), format="svg", bbox_inches="tight")
