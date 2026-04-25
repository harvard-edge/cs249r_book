import os
import matplotlib.pyplot as plt

time = [0, 5, 10, 15, 20]
checkpoints = [1, 1, 1, 1, 1]
plt.figure(figsize=(6, 2))
plt.stem(time, checkpoints, linefmt='#3d9e5a', markerfmt='D')
plt.yticks([])
plt.xlabel('Time (Minutes)')
plt.title('5-Minute Checkpointing Schedule')

out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')