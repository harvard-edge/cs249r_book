import os
import numpy as np
import matplotlib.pyplot as plt

# System parameters
mu = 50.0  # service rate (fps)
service_time_ms = (1.0 / mu) * 1000  # 20 ms

# Utilization
rho = np.linspace(0.01, 0.95, 100)

# M/D/1 Wait time: Wq = rho / (2*mu*(1-rho)) * 1000
# Total Latency = Wq + service_time
latency_md1 = (rho / (2 * mu * (1 - rho))) * 1000 + service_time_ms

# Bursty arrivals approximation (Kingman's formula)
# Ca^2 (arrival variance) = 5.0, Cs^2 (deterministic service) = 0
Ca2 = 5.0
latency_burst = ((Ca2 + 0) / 2) * (rho / (1 - rho)) * service_time_ms + service_time_ms

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(rho, latency_md1, label='Poisson Arrivals (M/D/1)', color='#4a90c4', linewidth=2)
ax.plot(rho, latency_burst, label='Bursty Arrivals (High Variance)', color='#c87b2a', linewidth=2, linestyle='--')

ax.set_title('Queueing Latency vs. Utilization', fontsize=14, pad=15)
ax.set_xlabel('Average Utilization (\u03c1 = \u03bb/\u03bc)', fontsize=12)
ax.set_ylabel('Expected Latency (ms)', fontsize=12)
ax.set_ylim(0, 300)
ax.axvline(x=0.8, color='grey', linestyle=':', label='Operating Point (\u03c1=0.8)')

# Annotate the specific operating point difference
ax.scatter([0.8], [(0.8 / (2 * 50 * 0.2)) * 1000 + 20], color='#4a90c4', zorder=5)
ax.scatter([0.8], [((5.0 / 2) * (0.8 / 0.2) * 20 + 20)], color='#c87b2a', zorder=5)

ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(fontsize=11, loc='upper left')

plt.tight_layout()
out_path = os.environ.get("VISUAL_OUT_PATH", "plot.svg")
plt.savefig(out_path, format="svg", bbox_inches="tight")
plt.close()