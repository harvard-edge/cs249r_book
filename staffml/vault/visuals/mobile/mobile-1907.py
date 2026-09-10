import os
import matplotlib.pyplot as plt
import numpy as np

minutes = np.linspace(0, 60, 60)
gb_written = (minutes * 60 * 30) / 1000

plt.figure(figsize=(6, 4))
plt.plot(minutes, gb_written, color='#c87b2a', lw=2)
plt.axhline(100, color='red', linestyle='--', label='100 GB Mark')
plt.xlabel('Time (Minutes)')
plt.ylabel('Data Written (GB)')
plt.legend()

out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')