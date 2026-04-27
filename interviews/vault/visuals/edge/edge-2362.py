import os
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 4))
ax.barh(['Static Max', 'Paged KV'], [100, 15], color=['#fdebd0', '#cfe2f3'], edgecolor=['#c87b2a', '#4a90c4'])
ax.set_xlabel('Memory Allocated per Request (%)')
ax.set_title('KV Cache Memory Waste Avoidance')
plt.savefig(os.environ['VISUAL_OUT_PATH'], format='svg', bbox_inches='tight')