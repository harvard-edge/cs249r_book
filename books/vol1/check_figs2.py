import re
with open("05_nn_computation/05_nn_computation.qmd", "r") as f:
    text = f.read()

figs = re.findall(r'::: \{#fig-([^ ]+)', text)
refs = re.findall(r'@[Ff]ig-([a-zA-Z0-9\-]+)', text)

print("Not in refs:")
for fig in set(figs):
    if fig not in refs:
        print(fig)
