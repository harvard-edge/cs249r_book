import re

file_path = "/Users/VJ/GitHub/MLSysBook-gem-vol3/publishing/quarto/contents/vol3/draft_v2/05_episodic_memory/05_episodic_memory.qmd"
with open(file_path, "r") as f:
    content = f.read()

# Replace block 1
block1_regex = re.compile(r"<!-- MERMAID SOURCE PRESERVATION.*?fig-mermaid-05_episodic_memory-0\.png\)\{#fig-mermaid-05_episodic_memory-0\}", re.DOTALL)
content = block1_regex.sub("![**Figure 5.1: The Agent Memory and Storage Hierarchy**](/assets/images/svg/vol3/05-episodic-memory-stack.svg){#fig-vol3-episodic-memory-stack}", content)

# Replace block 2
block2_regex = re.compile(r"<!-- MERMAID SOURCE PRESERVATION.*?fig-mermaid-05_episodic_memory-1\.png\)\{#fig-mermaid-05_episodic_memory-1\}", re.DOTALL)
content = block2_regex.sub("![**Figure 5.2: Write-Ahead Logging and Transactional Actuation Boundary**](/assets/images/svg/vol3/05-episodic-wal-flow.svg){#fig-vol3-episodic-wal-flow}", content)

# Replace block 3
block3_regex = re.compile(r"<!-- MERMAID SOURCE PRESERVATION.*?fig-mermaid-05_episodic_memory-2\.png\)\{#fig-mermaid-05_episodic_memory-2\}", re.DOTALL)
content = block3_regex.sub("![**Figure 5.3: IVF-PQ Structural Pipeline and Asymmetric Distance Computation**](/assets/images/svg/vol3/05-episodic-ivf-pq.svg){#fig-vol3-episodic-ivf-pq}", content)

with open(file_path, "w") as f:
    f.write(content)

