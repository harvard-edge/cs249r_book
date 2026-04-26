import os
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot([1, 2, 3, 4, 5], [1, 2, 3, 4, 4], label='GPipe', color='#4a90c4', marker='o')
ax.plot([1, 2, 3, 4, 5], [1, 2, 2, 2, 2], label='1F1B', color='#3d9e5a', marker='x')
ax.set_xlabel('Time Step')
ax.set_ylabel('Active Microbatches in Memory')
ax.set_title('Peak Memory Accumulation')
ax.legend()
plt.savefig(os.environ['VISUAL_OUT_PATH'], format='svg', bbox_inches='tight')