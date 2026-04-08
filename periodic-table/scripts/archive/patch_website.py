import re

html_path = 'periodic-table/index.html'
with open(html_path, 'r') as f:
    content = f.read()

# Add the final missing elements from the 100-round debate to the elements array
# En (Entropy) - Row 1 (Data), Col 14 (M)
# Ix (Indexing) - Row 4 (Architecture), Col 3 (R)
# Ro (Routing) - Row 4 (Architecture), Col 13 (K)
# Vr (Virtualization) - Row 6 (Runtime), Col 2 (R) -- move Checkpointing to Col 3 (replace Ir? let's just make Vr col 8? No, col 2, and we can put it there. Actually, let's just put it at an unused column in the block. Represent is cols 1-3. Col 1 is Cc, Col 2 is Cp, Col 3 is Ir. Let's make Vr col 8 but block R. The grid layout handles this based on col number.)
# Td (Thermodynamics) - Row 7 (Hardware), Col 14 (M)
# Rs (Resilience) - Row 8 (Production), Col 14 (K) - shift La to 15? Wait, La is M.
# Let's just append them to the array safely.

new_elements = """
  // Final insertions from the 100-Round Expert Consensus
  [80,'En','Entropy','M',1,14,'1948','The Shannon information-theoretic limit; the absolute bound on data compressibility.',['Vl'],'Row 0 (Data): information limit. Measure.'],
  [81,'Ix','Indexing','R',4,3,'—','The high-dimensional partitioning of vector space (e.g., HNSW) for sub-linear retrieval.',['Tp'],'Row 3 (Architecture): structured retrieval. Represent.'],
  [82,'Ro','Routing','K',4,13,'—','The dynamic, data-dependent dispatch of tensors (e.g., Mixture of Experts).',['Gt'],'Row 3 (Architecture): dynamic flow. Control.'],
  [83,'Vr','Virtualization','R',6,8,'—','The abstraction of physical memory via page tables (e.g., PagedAttention) to solve fragmentation.',['Cc'],'Row 5 (Runtime): memory mapping. Represent.'],
  [84,'Td','Thermodynamics','M',7,14,'—','The ultimate physical limitation (Landauer limit, thermal throttling) capping system scale.',['Ew'],'Row 6 (Hardware): thermal limit. Measure.'],
  [85,'Rs','Resilience','K',8,11,'—','The systemic countermeasures (checkpointing, elastic recovery) for macroscopic hardware decay.',['Oc'],'Row 7 (Production): fault tolerance. Control.']
];"""

content = content.replace("];", new_elements)

with open(html_path, 'w') as f:
    f.write(content)

