#!/usr/bin/env python3
"""Float-explanation audit scanner (flag-only, no edits).

Enumerates floats in a chapter .qmd and assembles, for each one, the CONTEXT BUNDLE a
human/judging pass needs to rule on whether the float is *explained* in its neighborhood
(comprehension), not merely whether the ref resolves (covered by other tools).

THE FLAG RULE this serves: a float is under-explained only if the explanation is absent
from the WHOLE local neighborhood — the reference sentence, the paragraph it sits in, the
paragraph before, the paragraph(s) after, the caption, AND the prose right after the float.
The scanner gathers that neighborhood; it does NOT judge and does NOT edit.

Float types and detection:
  figure    def `#fig-`         ref @fig-/@Fig-
  table     def `#tbl-`         ref @tbl-/@Tbl-
  listing   def `#lst-`         ref @lst-/@Lst-
  equation  def `#eq-`          ref @eq-/@Eq-
  algorithm def `label: algo-`  ref @algo-/@Algo-/@alg-/@Alg-   (lives inside a code fence)

Usage:
  python3 scan_floats.py <chapter.qmd> [--types fig,tbl,lst,alg,eq] [--format bundle|md|json]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

TYPES = {"fig": "figure", "tbl": "table", "lst": "listing", "alg": "algorithm", "eq": "equation"}

# Label names: internal hyphens/underscores only — never absorb a trailing ':' or '.'
# (a ref like `@eq-foo:` or `@eq-foo.` must match the `eq-foo` definition).
_NAME = r"[A-Za-z0-9_]+(?:-[A-Za-z0-9_]+)*"
# div / inline-label defs (#fig- #tbl- #lst- #eq-) and algorithm `label: algo-...`
DEF_RE = re.compile(r"#(fig|tbl|lst|eq)-(" + _NAME + r")")
ALG_DEF_RE = re.compile(r"\blabel:\s*(algo?)-(" + _NAME + r")")
REF_RE = re.compile(r"@([Ff]ig|[Tt]bl|[Ll]st|[Aa]lgo|[Aa]lg|[Ee]q)-(" + _NAME + r")")
CAP_RE = re.compile(r'(?:fig-cap|lst-cap|tbl-cap)="((?:[^"\\]|\\.)*)"')
LATEX_CAP_RE = re.compile(r"\\caption\{((?:[^{}]|\{[^{}]*\})*)\}")
# Markdown pipe-table caption line: ": **Title**: desc {#tbl-foo}"
TBL_CAP_RE = re.compile(r"^\s*:\s+(.*?)\s*\{#tbl-")


def norm_type(prefix: str) -> str:
    p = prefix.lower()
    if p in ("alg", "algo"):
        return "alg"
    return p


def blocks(lines: list[str]) -> list[tuple[int, int, str]]:
    """Maximal runs of non-blank lines -> (start_line_1based, end_line_1based, text)."""
    out, i, n = [], 0, len(lines)
    while i < n:
        if lines[i].strip() == "":
            i += 1
            continue
        j = i
        while j < n and lines[j].strip() != "":
            j += 1
        out.append((i + 1, j, "\n".join(lines[i:j])))
        i = j
    return out


def block_at(blks: list[tuple[int, int, str]], line: int) -> int:
    for idx, (s, e, _) in enumerate(blks):
        if s <= line <= e:
            return idx
    return -1


PROSE_START = re.compile(r'^\s*[A-Z@$"\[]')
NONPROSE_START = re.compile(r"^\s*(```|:::|\||#\||#{1,6}\s|<|!\[|\$\$|\\(begin|end|caption|Require|Ensure|State))")


def first_prose_after(blks: list[tuple[int, int, str]], after_line: int) -> str:
    for s, e, text in blks:
        if s <= after_line:
            continue
        first = text.splitlines()[0]
        if NONPROSE_START.match(first):
            continue
        if PROSE_START.match(first):
            return f"[L{s}-{e}] {text}"
    return ""


def float_end(lines: list[str], def_idx0: int, kind: str) -> int:
    """1-based line where the float's structural body ends, for locating the payoff."""
    n = len(lines)
    line = lines[def_idx0]
    if kind in ("fig", "lst") and ":::" in line and "{#" in line:
        for k in range(def_idx0 + 1, n):
            if lines[k].strip() == ":::":
                return k + 1
    if kind == "alg":  # label sits inside a ``` fence; end at the next fence close
        for k in range(def_idx0 + 1, n):
            if lines[k].lstrip().startswith("```"):
                return k + 1
    return def_idx0 + 1


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z@$\[])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def ref_sentence(line: str) -> str:
    sents = split_sentences(line)
    hit = [s for s in sents if REF_RE.search(s)]
    return " ".join(hit) if hit else line.strip()


def in_code_fence(lines: list[str], idx0: int) -> bool:
    fence = 0
    for i in range(idx0):
        if lines[i].lstrip().startswith("```"):
            fence ^= 1
    return bool(fence)


def scan(path: Path, want: set[str]) -> dict:
    lines = path.read_text(encoding="utf-8").splitlines()
    blks = blocks(lines)

    defs: dict[str, dict] = {}

    def add_def(key, kind, line1, caption):
        if want and kind not in want:
            return
        defs.setdefault(key, {"type": TYPES[kind], "kind": kind, "def_line": line1, "caption": caption})

    for i, line in enumerate(lines):
        for m in DEF_RE.finditer(line):
            kind = norm_type(m.group(1))
            cap = CAP_RE.search(line)
            caption = cap.group(1) if cap else ""
            if not caption and kind == "tbl":
                tc = TBL_CAP_RE.search(line)  # markdown pipe-table ": ... {#tbl-}" caption
                caption = tc.group(1) if tc else ""
            add_def(f"{kind}-{m.group(2)}", kind, i + 1, caption)
        for m in ALG_DEF_RE.finditer(line):
            # caption is on a nearby \caption{...} line within the fence
            cap = ""
            for k in range(i, min(i + 12, len(lines))):
                lc = LATEX_CAP_RE.search(lines[k])
                if lc:
                    cap = lc.group(1)
                    break
            add_def(f"alg-{m.group(2)}", "alg", i + 1, cap)

    # references
    refs: list[dict] = []
    for i, line in enumerate(lines):
        for m in REF_RE.finditer(line):
            kind = norm_type(m.group(1))
            if want and kind not in want:
                continue
            refs.append({
                "key": f"{kind}-{m.group(2)}",
                "raw": m.group(0),
                "line": i + 1,
                "sentence": ref_sentence(line),
                "in_code": in_code_fence(lines, i),
            })

    def_keys = set(defs)
    # attach context bundle per def
    for key, d in defs.items():
        d_refs = [r for r in refs if r["key"] == key and not r["in_code"]]
        d["code_ref_count"] = sum(1 for r in refs if r["key"] == key and r["in_code"])
        bundle_refs = []
        for r in d_refs:
            bi = block_at(blks, r["line"])
            prev_b = f"[L{blks[bi-1][0]}-{blks[bi-1][1]}] {blks[bi-1][2]}" if bi > 0 else ""
            this_b = f"[L{blks[bi][0]}-{blks[bi][1]}] {blks[bi][2]}" if bi >= 0 else r["sentence"]
            next_b = f"[L{blks[bi+1][0]}-{blks[bi+1][1]}] {blks[bi+1][2]}" if 0 <= bi < len(blks) - 1 else ""
            bundle_refs.append({**r, "prev": prev_b, "this": this_b, "next": next_b})
        end = float_end(lines, d["def_line"] - 1, d["kind"])
        d["payoff"] = first_prose_after(blks, end)
        d["refs"] = bundle_refs
        d["orphan"] = len(d_refs) == 0
        if d["orphan"]:
            # bundle the prose around the definition: sometimes described but @ref forgotten
            d["near_def"] = first_prose_after(blks, end)

    dangling = [r for r in refs if r["key"] not in def_keys and not r["in_code"]]
    return {"defs": defs, "dangling": dangling}


def render_bundle(path: Path, result: dict) -> str:
    out = [f"# Float-explanation CONTEXT BUNDLE — `{path.name}`", ""]
    out.append("> For each float: caption, every prose reference with its PREV / THIS / NEXT paragraph,")
    out.append("> and the PAYOFF paragraph right after the float. Judge against ALL of it before flagging.")
    out.append("")
    by_type: dict[str, list] = {}
    for key, d in result["defs"].items():
        by_type.setdefault(d["type"], []).append((key, d))
    for t in sorted(by_type):
        items = sorted(by_type[t], key=lambda x: x[1]["def_line"])
        out.append(f"## {t.capitalize()}s ({len(items)})")
        out.append("")
        for key, d in items:
            out.append(f"### `{key}` — def L{d['def_line']}")
            out.append(f"- **Caption:** {d['caption'] or '(none found)'}")
            if d["orphan"]:
                out.append("- **Refs:** 🛑 NONE (orphan — check whether nearby prose describes it anyway)")
                if d.get("near_def"):
                    out.append(f"  - _Prose after float:_ {d['near_def']}")
            else:
                out.append(f"- **Refs ({len(d['refs'])}):**")
                for r in d["refs"]:
                    out.append(f"  - **L{r['line']} `{r['raw']}`**")
                    if r["prev"]:
                        out.append(f"    - prev ¶: {r['prev']}")
                    out.append(f"    - this ¶: {r['this']}")
                    if r["next"]:
                        out.append(f"    - next ¶: {r['next']}")
            if d.get("payoff"):
                out.append(f"- **Payoff ¶ (after float):** {d['payoff']}")
            if d["code_ref_count"]:
                out.append(f"- _(+{d['code_ref_count']} mention(s) inside code — not prose)_")
            out.append("")
    if result["dangling"]:
        out.append("## ⚠️ Dangling refs (no matching def)")
        for r in result["dangling"]:
            out.append(f"- L{r['line']} `{r['raw']}`: {r['sentence']}")
    return "\n".join(out)


def render_md(path: Path, result: dict) -> str:
    # compact inventory (counts only)
    by_type: dict[str, int] = {}
    orph: dict[str, int] = {}
    for d in result["defs"].values():
        by_type[d["type"]] = by_type.get(d["type"], 0) + 1
        if d["orphan"]:
            orph[d["type"]] = orph.get(d["type"], 0) + 1
    out = [f"# Float inventory — `{path.name}`", ""]
    for t in sorted(by_type):
        out.append(f"- {t}: {by_type[t]} defs, {orph.get(t,0)} orphan")
    if result["dangling"]:
        out.append(f"- dangling refs: {len(result['dangling'])}")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("chapter", type=Path)
    ap.add_argument("--types", default="")
    ap.add_argument("--format", choices=["bundle", "md", "json"], default="bundle")
    args = ap.parse_args()

    want = {norm_type(t) for t in (x.strip() for x in args.types.split(",")) if t}
    result = scan(args.chapter, want)
    if args.format == "bundle":
        print(render_bundle(args.chapter, result))
    elif args.format == "md":
        print(render_md(args.chapter, result))
    else:
        print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
