"""Write the student-notes variant of a deck: activity frames print before the answer.

Usage: python3 tools/student_source.py <deck.tex> <out.tex>

Handouts print each frame in its final build state, so a Predict or Recall
slide would show its revealed answer above the question. For the student notes
build (make student-notes), inside every frame that carries \\mlsysactivity,
each overlay specification that starts at the frame's answer step gains
"|handout:0", so the printed page shows the question and its choices but not
the answer, which the notes below give. The answer step is the first
highlighted choice (\\alt<k->{\\colorbox...} or a deck answer macro called with
step k), else the frame's last step. Build-step figures (\\mlsyssteps) print
their first step. (Handout mode otherwise treats a
plain <2-> as shown.) Specifications that already name a mode are left alone.
The deck source, the projected PDF, and the instructor notes are unchanged.
(2026-11-23)
"""
import re
import sys
from pathlib import Path

FRAME = re.compile(r"\\begin\{frame\}")


SPEC = re.compile(r"<(\d+)(-\d*)?>")


HIGHLIGHT = re.compile(r"\\alt<(\d+)-?>\s*\{\s*\\colorbox")


REVEAL = re.compile(r"\\(?:uncover|only|onslide|visible)<(\d+)-?>\s*\{")


def choice_steps(body: str) -> list:
    """Steps at which choices (\\choice, \\item) are revealed rather than shown from the start."""
    out = []
    for m in REVEAL.finditer(body):
        end = balanced(body, m.end() - 1)
        if re.search(r"\\(choice|item)\b", body[m.end():end]):
            out.append(int(m.group(1)))
    return out


def answer_step(body: str, names: list) -> int:
    """The step at which an activity frame reveals its answer.

    If every choice is visible from the start, any later reveal is an answer or
    explanation: step 2. Otherwise the first highlighted choice
    (\\alt<k->{\\colorbox...} or a deck answer macro called with step k), else the
    frame's last step, and always after the last revealed choice."""
    revealed = choice_steps(body)
    if not revealed:
        return 2
    ks = [int(k) for k in HIGHLIGHT.findall(body)]
    for name in names:
        ks += [int(k) for k in re.findall(r"\\" + re.escape(name) + r"\{(\d+)\}", body)]
    steps = [int(m.group(1)) for m in SPEC.finditer(body)]
    k = min(ks) if ks else (max(steps) if steps else 0)
    return max(k, max(revealed) + 1)


def hide_reveals(body: str, threshold: int) -> tuple[str, int]:
    count = 0

    def sub(m):
        nonlocal count
        if threshold >= 2 and int(m.group(1)) >= threshold:
            count += 1
            return f"<{m.group(1)}{m.group(2) or ''}|handout:0>"
        return m.group(0)

    return SPEC.sub(sub, body), count


STEPS = re.compile(r"\\mlsyssteps(\[[^\]]*\])?\{([^}]*)\}\{\d+\}")
DEF = re.compile(r"\\(?:re)?newcommand\{?\\([A-Za-z]+)\}?\[(\d)\]")


def balanced(text: str, i: int) -> int:
    """Index just past the brace group starting at text[i] == '{'."""
    depth = 0
    while i < len(text):
        c = text[i]
        if c == "\\":
            i += 2
            continue
        depth += {"{": 1, "}": -1}.get(c, 0)
        i += 1
        if depth == 0:
            return i
    return i


def macro_twins(text: str):
    """Preamble macros whose body reveals at a parameter step (<#1->, <#2>):
    return (twin definitions, names). The twin hides those reveals in handouts."""
    twins, names = [], []
    head = text[: text.find("\\begin{document}")] if "\\begin{document}" in text else text
    for m in DEF.finditer(head):
        start = head.find("{", m.end())
        if start < 0:
            continue
        end = balanced(head, start)
        body = head[start:end]
        if not re.search(r"<#\d", body):
            continue
        twin_body = re.sub(r"<(#\d)(-?)>", r"<\1\2|handout:0>", body)
        name = m.group(1)
        twins.append(f"\\newcommand{{\\{name}Mlsysact}}[{m.group(2)}]{twin_body}")
        names.append(name)
    return twins, names


def transform(text: str) -> tuple[str, int, int]:
    twins, names = macro_twins(text)
    call = re.compile(r"\\(" + "|".join(map(re.escape, names)) + r")(?![A-Za-z])") if names else None
    out, pos, frames, specs = [], 0, 0, 0
    for m in FRAME.finditer(text):
        s = m.start()
        e = text.find("\\end{frame}", s)
        e = len(text) if e < 0 else e
        out.append(text[pos:s])
        body = text[s:e]
        if "\\mlsysactivity" in body:
            notes_at = body.find("\\note{")
            slide, notes = (body[:notes_at], body[notes_at:]) if notes_at >= 0 else (body, "")
            k = answer_step(slide, names)
            slide, n = hide_reveals(slide, k)
            body = slide + notes
            body, kk = STEPS.subn(lambda mm: f"\\includegraphics{mm.group(1) or '[width=\\textwidth,keepaspectratio]'}{{{mm.group(2)}1}}", body)
            c = 0
            if call:
                def twin(mm):
                    nonlocal c
                    arg = re.match(r"\{(\d+)\}", body[mm.end():mm.end() + 6])
                    if arg and k >= 2 and int(arg.group(1)) >= k:
                        c += 1
                        return f"\\{mm.group(1)}Mlsysact"
                    return mm.group(0)
                body = call.sub(twin, body)
            frames += 1
            specs += n + kk + c
        out.append(body)
        pos = e
    out.append(text[pos:])
    new = "".join(out)
    if twins:
        i = new.find("\\begin{document}")
        if i >= 0:
            new = new[:i] + "\n".join(twins) + "\n" + new[i:]
    return new, frames, specs


def residual(text: str) -> list:
    """Reveals left on activity frames that this transform cannot see (loop
    variables such as <\\k->): reported, not edited."""
    left = []
    for m in FRAME.finditer(text):
        e = text.find("\\end{frame}", m.start())
        body = text[m.start(): e if e >= 0 else len(text)]
        if "\\mlsysactivity" in body:
            left += re.findall(r"<\\[A-Za-z]+-?>", body)
    return left


if __name__ == "__main__":
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    new, frames, specs = transform(src.read_text())
    dst.write_text(new)
    left = residual(new)
    print(f"{src.name}: {frames} activity frames, {specs} reveals hidden in the student notes"
          + (f"; {len(left)} loop-variable reveals left: {sorted(set(left))}" if left else ""))
