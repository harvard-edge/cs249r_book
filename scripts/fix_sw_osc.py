with open("books/vol3/11_scheduling/images/svg/semantic-watchdog-oscillation.svg") as f:
    txt = f.read()

import re

# replace arc and pill
old = re.search(r'<path d="M 350 165.*?<text x="570" y="330".*?</text>', txt, re.DOTALL).group(0)

new = """<path d="M 360 245 C 360 320 800 320 800 252" class="edge-alert" marker-end="url(#arrow-red)" />
  
  <rect x="420" y="300" width="300" height="52" class="pill-alert" />
  <text x="570" y="322" class="mono-alert">HASH COLLISION: h(s_t) == h(s_{t-2})</text>
  <text x="570" y="340" class="annot-text" text-anchor="middle" fill="#dc2626">Environment reverted without forward progress (Orbit = 2)</text>"""

txt = txt.replace(old, new)
with open("books/vol3/11_scheduling/images/svg/semantic-watchdog-oscillation.svg", "w") as f:
    f.write(txt)

