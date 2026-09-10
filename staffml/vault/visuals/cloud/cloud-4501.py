import os
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0, 10, 50)
y_uncontrolled = np.exp(x * 0.4)
y_controlled = np.clip(y_uncontrolled, 0, 15)
plt.plot(x, y_uncontrolled, color='#c87b2a', linestyle='--', label='Uncontrolled')
plt.plot(x, y_controlled, color='#3d9e5a', label='Admit Cap')
plt.legend()
plt.ylabel('Queue Length')
plt.xlabel('Time')
out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')