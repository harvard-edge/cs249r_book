import os
import matplotlib.pyplot as plt

time = [0, 5, 10]
queue_size = [0, 50, 100]

plt.figure(figsize=(6, 4))
plt.plot(time, queue_size, color='#c87b2a', lw=2)
plt.xlabel('Time Overloaded (Seconds)')
plt.ylabel('Queue Size')
plt.title('Arrival Rate > Service Rate')

out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')