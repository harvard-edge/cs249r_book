import re

with open('/Users/VJ/GitHub/MLSysBook-gem-vol3/publishing/quarto/contents/vol3/draft_v2/04_virtual_memory/04_virtual_memory.qmd', 'r') as f:
    content = f.read()

# Replace block 1
block1_regex = r"<!-- MERMAID SOURCE PRESERVATION\nflowchart TD\n    subgraph T1.*?-->\n!\[Virtual context memory as the foundational physical execution substrate in the autonomous agent systems stack\.\]\(/assets/images/diagrams/vol3/fig-vol3-vm-agent-stack\.png\)\{#fig-vol3-vm-agent-stack\}"
replacement1 = "![**Figure 4.1: Virtual context memory as the foundational physical execution substrate in the autonomous agent systems stack.**](/assets/images/svg/vol3/fig-vol3-vm-agent-stack.svg){#fig-vol3-vm-agent-stack}"
content = re.sub(block1_regex, replacement1, content, flags=re.DOTALL)

# Replace block 2
block2_regex = r"```\{mermaid\}\n%%\| label: fig-vol3-vm-prefix-locality\n%%\| fig-cap: \"Prefix locality and append-only expansion across three consecutive turns of an agent tool-use trajectory\.\"\n%%\| fig-alt: \".*?\"\n.*?\n```"
replacement2 = "![**Figure 4.2: Prefix locality and append-only expansion across three consecutive turns of an agent tool-use trajectory.**](/assets/images/svg/vol3/fig-vol3-vm-prefix-locality.svg){#fig-vol3-vm-prefix-locality}"
content = re.sub(block2_regex, replacement2, content, flags=re.DOTALL)

# Replace block 3
block3_regex = r"<!-- MERMAID SOURCE PRESERVATION\nflowchart TD\n    ROOT.*?-->\n!\[Physical block sharing across a 3-branch speculative search tree.*?\]\(/assets/images/diagrams/vol3/fig-vol3-vm-tree-kv-cache\.png\)\{#fig-vol3-vm-tree-kv-cache\}"
replacement3 = "![**Figure 4.3: Physical block sharing across a 3-branch speculative search tree.**](/assets/images/svg/vol3/fig-vol3-vm-tree-kv-cache.svg){#fig-vol3-vm-tree-kv-cache}"
content = re.sub(block3_regex, replacement3, content, flags=re.DOTALL)

with open('/Users/VJ/GitHub/MLSysBook-gem-vol3/publishing/quarto/contents/vol3/draft_v2/04_virtual_memory/04_virtual_memory.qmd', 'w') as f:
    f.write(content)
print("Replaced!")
