import os
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 4))
tiers = ['CPU DRAM\n(PCIe Gen5)', 'Effective Target', 'HBM3 (H100)']
bw = [64, 1600, 3200]
ax.bar(tiers, bw, color=['#cfe2f3', '#fdebd0', '#d4edda'], edgecolor='black')
ax.set_ylabel('Bandwidth (GB/s)')
ax.set_title('Memory Tier Bandwidth Comparison')
ax.set_yscale('log')
plt.axhline(y=1600, color='r', linestyle='--')
plt.tight_layout()
out = os.environ.get("VISUAL_OUT_PATH", "out.svg")
plt.savefig(out, format="svg", bbox_inches="tight")