import os
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 4))
stages = 4
microbatches = 8
colors = ['#cfe2f3', '#d4edda']
for s in range(stages):
    for m in range(microbatches):
        ax.barh(stages - 1 - s, 0.8, left=s+m, height=0.6, color=colors[0], edgecolor='black')
    for m in range(microbatches):
        ax.barh(stages - 1 - s, 0.8, left=stages+s+m+microbatches-2, height=0.6, color=colors[1], edgecolor='black')
ax.set_yticks(range(stages))
ax.set_yticklabels([f'Stage {i}' for i in reversed(range(stages))])
ax.set_xlabel('Time Steps')
ax.set_title('1F1B Pipeline Schedule Visualization')
plt.tight_layout()
out = os.environ.get("VISUAL_OUT_PATH", "out.svg")
plt.savefig(out, format="svg", bbox_inches="tight")