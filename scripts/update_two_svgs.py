# 1. Update semantic-watchdog-oscillation.svg
with open("books/vol3/11_scheduling/images/svg/semantic-watchdog-oscillation.svg") as f:
    txt = f.read()

# Replace arc and pill
old_arc = '<path d="M 360 245 C 360 300 800 300 800 251" class="edge-alert" marker-end="url(#arrow-red)" />\n  \n  <rect x="430" y="295" width="280" height="48" class="pill-alert" />\n  <text x="570" y="315" class="mono-alert">HASH COLLISION: h(s_t) == h(s_{t-2})</text>\n  <text x="570" y="331" class="annot-text" text-anchor="middle" fill="#dc2626">Environment reverted without forward progress (Orbit = 2)</text>'

new_arc = '<path d="M 350 165 C 350 285 790 285 790 173" class="edge-alert" marker-end="url(#arrow-red)" />\n  \n  <rect x="420" y="290" width="300" height="52" class="pill-alert" />\n  <text x="570" y="312" class="mono-alert">HASH COLLISION: h(s_t) == h(s_{t-2})</text>\n  <text x="570" y="330" class="annot-text" text-anchor="middle" fill="#dc2626">Environment reverted without forward progress (Orbit = 2)</text>'

txt = txt.replace(old_arc, new_arc)
with open("books/vol3/11_scheduling/images/svg/semantic-watchdog-oscillation.svg", "w") as f:
    f.write(txt)

# 2. Update fig-vol3-fault-tolerant-synthesis.svg
with open("books/vol3/11_scheduling/images/svg/fig-vol3-fault-tolerant-synthesis.svg") as f:
    txt2 = f.read()

txt2 = txt2.replace('<text x="720" y="433" class="msg-mono" text-anchor="middle" fill="#b45309">10. exec_compensator(C2: git checkout auth.py)</text>',
                    '<text x="615" y="433" class="msg-mono" text-anchor="start" fill="#b45309">10. exec_compensator(C2: git checkout auth.py)</text>')
txt2 = txt2.replace('<text x="720" y="461" class="msg-mono" text-anchor="middle" fill="#b45309">11. exec_compensator(C1: git worktree remove -f)</text>',
                    '<text x="615" y="461" class="msg-mono" text-anchor="start" fill="#b45309">11. exec_compensator(C1: git worktree remove -f)</text>')

with open("books/vol3/11_scheduling/images/svg/fig-vol3-fault-tolerant-synthesis.svg", "w") as f:
    f.write(txt2)

print("Updated two SVGs")
