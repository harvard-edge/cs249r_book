import os
import matplotlib.pyplot as plt

page_sizes = ['256-Token Page', '16-Token Page']
max_waste = [255, 15]

plt.figure(figsize=(6, 3))
plt.bar(page_sizes, max_waste, color=['#c87b2a', '#3d9e5a'], width=0.5)
plt.ylabel('Max Wasted Tokens')
plt.title('Internal Fragmentation Reduction')
plt.savefig(os.environ.get('VISUAL_OUT_PATH', 'out.svg'), format='svg', bbox_inches='tight')