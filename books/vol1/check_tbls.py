import re

with open("05_nn_computation/05_nn_computation.qmd", "r") as f:
    text = f.read()

tbls = re.findall(r'::: \{#tbl-([^ ]+)', text)
refs = re.findall(r'@[Tt]bl-([a-zA-Z0-9\-]+)', text)

for tbl in set(tbls):
    if tbl not in refs:
        print(f"Orphaned Table: {tbl}")
