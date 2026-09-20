# 1. Fix fig-vol3-failure-diagnostic-tree.svg
with open("books/vol3/12_data_flywheel/images/svg/fig-vol3-failure-diagnostic-tree.svg") as f:
    txt = f.read()

txt = txt.replace('d="M 620 282 L 780 282 L 780 340"', 'd="M 620 282 L 780 282 L 780 338"')
with open("books/vol3/12_data_flywheel/images/svg/fig-vol3-failure-diagnostic-tree.svg", "w") as f:
    f.write(txt)

# 2. Fix vol3-collection-pipeline.svg
with open("books/vol3/12_data_flywheel/images/svg/vol3-collection-pipeline.svg") as f:
    txt2 = f.read()

txt2 = txt2.replace('<text x="80" y="170" class="mono" font-weight="700" fill="#166534">Object Store Sink</text>',
                    '<text x="80" y="170" class="mono" font-weight="700" fill="#166534" text-anchor="middle">Object Store Sink</text>')
txt2 = txt2.replace('<text x="80" y="225" class="mono" font-weight="700" fill="#dc2626">Quarantine Deadletter</text>',
                    '<text x="80" y="225" class="mono" font-weight="700" fill="#dc2626" text-anchor="middle">Quarantine Sink</text>')

with open("books/vol3/12_data_flywheel/images/svg/vol3-collection-pipeline.svg", "w") as f:
    f.write(txt2)

print("Fixed two SVGs")
