import re

file_path = "/Users/VJ/GitHub/MLSysBook-gem-vol3/publishing/quarto/contents/vol3/draft_v2/08_actuation/08_actuation.qmd"
with open(file_path, "r") as f:
    content = f.read()

pattern = re.compile(r"<!-- MERMAID SOURCE PRESERVATION.*?!\[Diagram\]\([^)]+\)\{#[^}]+\}", re.DOTALL)

matches = pattern.findall(content)
print(f"Found {len(matches)} matches")

# Replace each match with the corresponding SVG
svgs = [
    "![](/assets/images/svg/vol3/08_actuation_0.svg)",
    "![](/assets/images/svg/vol3/08_actuation_1.svg)",
    "![](/assets/images/svg/vol3/08_actuation_2.svg)",
    "![](/assets/images/svg/vol3/08_actuation_3.svg)",
    "![](/assets/images/svg/vol3/08_actuation_4.svg)"
]

for i, match in enumerate(matches):
    if i < len(svgs):
        content = content.replace(match, svgs[i])

with open(file_path, "w") as f:
    f.write(content)

print("Replaced all matches.")
