import os
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
t_points = []
p_points = []
for i in range(3):
    start = i * 10
    t_points.extend([max(0, start - 1e-5), start, start + 0.02, start + 0.02 + 1e-5])
    p_points.extend([0.01, 15, 15, 0.01])
    if i < 2:
        t_points.append((i+1)*10 - 1e-5)
        p_points.append(0.01)

plt.plot(t_points, p_points, color='#4a90c4', linewidth=2)
plt.fill_between(t_points, p_points, 1e-3, color='#cfe2f3', alpha=0.5)

plt.axhline(y=0.04, color='red', linestyle='--', label='40 $\mu$W Avg Budget')
plt.axhline(y=0.01, color='green', linestyle=':', label='10 $\mu$W Sleep')

plt.yscale('log')
plt.ylim(0.005, 30)
plt.yticks([0.01, 0.04, 0.1, 1, 15], ['10 $\mu$W', '40 $\mu$W', '0.1 mW', '1 mW', '15 mW'])

plt.title("Duty Cycle Power Timeline (T = 10s)")
plt.xlabel("Time (seconds)")
plt.ylabel("Power")
plt.grid(True, axis='y', alpha=0.3)
plt.legend(loc='center right')

out_path = os.environ.get("VISUAL_OUT_PATH", "out.svg")
plt.savefig(out_path, format="svg", bbox_inches="tight")
plt.close()