import re

html_path = 'periodic-table/index.html'
with open(html_path, 'r') as f:
    content = f.read()

informal_elements = """  // Informal symbols used in Compound Formulas
  [86,'Ac','Activation','C',3,8,'—','Non-linear functions (ReLU, GELU) providing expressive power.',['Dd'],'Row 2 (Algorithm): non-linear transform. Compute.'],
  [87,'St','State','R',2,3,'—','The mathematical representation of an environment or context (RL, SSMs).',['Ob'],'Row 1 (Math): contextual state. Represent.'],
  [88,'Re','Retrieve','X',5,11,'—','Fetching stored state or external knowledge (e.g., from a KV Cache or Vector DB).',['Hs'],'Row 4 (Optimization): state retrieval. Communicate.'],
  [89,'Wa','Weight Avg','C',5,8,'—','Averaging model weights across time or distributed workers (e.g., SWA, EMA).',['Pm'],'Row 4 (Optimization): parameter smoothing. Compute.'],
  [90,'Ct','Critic','K',3,11,'—','The value function evaluating the expected return of a state (Actor-Critic RL).',['St','Gd'],'Row 2 (Algorithm): evaluative model. Control.']
];"""

content = content.replace("fault tolerance. Control.']\n];", "fault tolerance. Control.'],\n" + informal_elements)

with open(html_path, 'w') as f:
    f.write(content)
