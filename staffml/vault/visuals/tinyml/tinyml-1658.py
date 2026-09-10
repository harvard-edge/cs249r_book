import os
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6,4))
ax.bar(['INT8 Model', 'INT16 Model'], [200, 400], color=['#d4edda', '#fdebd0'], edgecolor=['#3d9e5a', '#c87b2a'])
ax.axhline(256, color='red', linestyle='--', label='SRAM Limit (256KB)')
ax.set_ylabel('Memory Footprint (KB)')
ax.set_title('SRAM Capacity Limit Saturation')
ax.legend()
plt.savefig(os.environ['VISUAL_OUT_PATH'], format='svg', bbox_inches='tight')