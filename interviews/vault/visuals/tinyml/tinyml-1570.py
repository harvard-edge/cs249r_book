import os
import matplotlib.pyplot as plt
import numpy as np
labels = ['SRAM Cap.', 'Model Layout']
sram_avail = [256, 0]
model_acts = [0, 80]
model_weights_sram = [0, 176]
model_weights_flash = [0, 44]
fig, ax = plt.subplots()
ax.bar(labels, sram_avail, color='#d4edda', label='Empty SRAM')
ax.bar(labels, model_acts, color='#c87b2a', label='Activations')
ax.bar(labels, model_weights_sram, bottom=model_acts, color='#4a90c4', label='Weights (SRAM)')
ax.bar(labels, model_weights_flash, bottom=np.add(model_acts, model_weights_sram), color='#cfe2f3', label='Weights (Flash)')
ax.legend()
ax.set_ylabel('Memory (KB)')
out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')