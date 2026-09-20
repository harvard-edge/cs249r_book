with open("books/vol3/13_sft/images/svg/ch13-sft-evaluation-radar.svg") as f:
    txt = f.read()

import re

old_labels = """  <!-- Axis Titles with ample clearance -->
  <text x="320" y="115" class="axis-title">Task Completion (Pass@1 %)</text>
  <text x="455" y="255" class="axis-title" text-anchor="start">Syntactic Tool Validity (%)</text>
  <text x="320" y="430" class="axis-title">Trajectory Efficiency (1 / Tokens)</text>
  <text x="185" y="255" class="axis-title" text-anchor="end">General Capability Retention (%)</text>"""

new_labels = """  <!-- Axis Titles with clean clearance and centering -->
  <text x="320" y="115" class="axis-title">Task Completion (Pass@1 %)</text>
  
  <text x="535" y="264" class="axis-title" text-anchor="middle">Syntactic Tool</text>
  <text x="535" y="280" class="axis-title" text-anchor="middle">Validity (%)</text>

  <text x="320" y="435" class="axis-title">Trajectory Efficiency (1 / Tokens)</text>

  <text x="105" y="264" class="axis-title" text-anchor="middle">General Retention</text>
  <text x="105" y="280" class="axis-title" text-anchor="middle">(%)</text>"""

txt = txt.replace(old_labels, new_labels)
with open("books/vol3/13_sft/images/svg/ch13-sft-evaluation-radar.svg", "w") as f:
    f.write(txt)

