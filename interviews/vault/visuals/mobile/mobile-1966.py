import os
import matplotlib.pyplot as plt

precisions = ['FP16 (2 bytes)', 'INT8 (1 byte)']
context_len = [1, 2] # normalized

plt.figure(figsize=(6, 3))
plt.barh(precisions, context_len, color=['#cfe2f3', '#d4edda'], edgecolor=['#4a90c4', '#3d9e5a'])
plt.xlabel('Max Context Tokens (Normalized)')
plt.title('KV Cache Quantization Impact')
plt.savefig(os.environ.get('VISUAL_OUT_PATH', 'out.svg'), format='svg', bbox_inches='tight')