import os
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6,4))
ax.bar(['Sequential', 'Random Access'], [160, 30], color=['#cfe2f3', '#fdebd0'], edgecolor=['#4a90c4', '#c87b2a'])
ax.axhline(204.8, color='red', linestyle='--', label='Theoretical Peak (204.8)')
ax.set_ylabel('Effective Bandwidth (GB/s)')
ax.set_title('AGX Orin LPDDR5 Utilization')
ax.legend()
plt.savefig(os.environ['VISUAL_OUT_PATH'], format='svg', bbox_inches='tight')