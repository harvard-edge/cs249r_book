import os
import matplotlib.pyplot as plt
import numpy as np
t = np.linspace(0, 180, 500)
power = np.where((t % 60) < 2, 33, 0) # Exaggerated width for visibility
plt.plot(t, power, color='#c87b2a')
plt.xlabel('Time (Seconds)')
plt.ylabel('Power (mW)')
plt.title('EEPROM Write Pulses')
out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')