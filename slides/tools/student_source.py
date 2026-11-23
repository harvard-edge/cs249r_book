"""Write the student-notes variant of a deck: activity frames print before the answer.

Usage: python3 tools/student_source.py <deck.tex> <out.tex>

Handouts print each frame in its final build state, so a Predict or Recall
slide would show its revealed answer above the question. For the student notes
build (make student-notes), inside every frame that carries \\mlsysactivity,
each overlay specification that starts at step 2 or later (<2->, <3>, <2-4>)
gains "|handout:0", so the printed page shows the slide before the answer
appears and the notes below give the answer. (Handout mode otherwise treats a
plain <2-> as shown.) Specifications that already name a mode are left alone.
The deck source, the projected PDF, and the instructor notes are unchanged.
(2026-11-23)
"""
import re
import sys
from pathlib import Path

FRAME = re.compile(r"\\begin\{frame\}")


SPEC = re.compile(r"<(\d+)(-\d*)?>")


def hide_reveals(body: str) -> tuple[str, int]:
    count = 0

    def sub(m):
        nonlocal count
        if int(m.group(1)) >= 2:
            count += 1
            return f"<{m.group(1)}{m.group(2) or ''}|handout:0>"
        return m.group(0)

    return SPEC.sub(sub, body), count


def transform(text: str) -> tuple[str, int, int]:
    out, pos, frames, specs = [], 0, 0, 0
    for m in FRAME.finditer(text):
        s = m.start()
        e = text.find("\\end{frame}", s)
        e = len(text) if e < 0 else e
        out.append(text[pos:s])
        body = text[s:e]
        if "\\mlsysactivity" in body:
            body, n = hide_reveals(body)
            frames += 1
            specs += n
        out.append(body)
        pos = e
    out.append(text[pos:])
    return "".join(out), frames, specs


if __name__ == "__main__":
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    new, frames, specs = transform(src.read_text())
    dst.write_text(new)
    print(f"{src.name}: {frames} activity frames, {specs} reveals hidden in the student notes")
