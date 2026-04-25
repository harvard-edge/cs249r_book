import os
import matplotlib.pyplot as plt
import numpy as np
mu = 150
lambda_vals = np.linspace(0, 140, 100)
latency = 1 / (mu - lambda_vals) * 1000
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(lambda_vals, latency, color='#4a90c4', linewidth=2)
ax.scatter([50, 125], [10, 40], color='#c87b2a', zorder=5)
ax.annotate('10ms @ 50 req/s', (50, 10), xytext=(20, 10), textcoords='offset points')
ax.annotate('40ms @ 125 req/s', (125, 40), xytext=(-90, 10), textcoords='offset points')
ax.set_xlabel('Arrival Rate (req/sec)')
ax.set_ylabel('Average Latency (ms)')
ax.set_title('M/M/1 Queue Latency Hockey-Stick')
ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
out = os.environ.get("VISUAL_OUT_PATH", "out.svg")
plt.savefig(out, format="svg", bbox_inches="tight")