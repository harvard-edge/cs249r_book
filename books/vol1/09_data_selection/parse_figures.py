import re

with open('09_data_selection.qmd', 'r') as f:
    text = f.read()

# Find figure definitions: e.g. {#fig-something} or label="fig-something" or similar
# Also standard markdown ![caption](path){#fig-id}
fig_defs = re.findall(r'\{#fig-([^}]+)\}', text)
# Find references: @fig-something
fig_refs = re.findall(r'@fig-([a-zA-Z0-9_-]+)', text)

print("Figure Definitions found:", len(set(fig_defs)))
for fd in set(fig_defs):
    count = fig_refs.count(fd)
    print(f" - {fd}: referenced {count} times")

# Also check for tables {#tbl-...} and @tbl-...
tbl_defs = re.findall(r'\{#tbl-([^}]+)\}', text)
tbl_refs = re.findall(r'@tbl-([a-zA-Z0-9_-]+)', text)
print("\nTable Definitions found:", len(set(tbl_defs)))
for td in set(tbl_defs):
    count = tbl_refs.count(td)
    print(f" - {td}: referenced {count} times")

