#!/usr/bin/env python3
"""
Body-prose emphasis extractor (READ-ONLY candidate generator).

Emits every **bold** and *italic* span that lives in BODY PROSE, deterministically
skipping anything the body-prose emphasis rules do not own:

  - YAML frontmatter
  - fenced code blocks  (``` ... ```  and  ~~~ ... ~~~ ,  incl. {python} cells)
  - inline code  `...`   and math  $...$ / $$...$$   (masked before span detection)
  - CALLOUT interiors    (::: {.callout-*} ... :::)   <-- the whole block, deferred to its own pass
  - HTML comments
  - headings (#...), table rows (| ... |), and caption/attribute lines
    (fig-cap= / tbl-cap= / lst-cap= / fig-alt= / tbl-alt= / title= inside {...})

Spans are tagged with a ZONE so non-body streams stay separable rather than silently dropped:
  body      - running body prose (the audit target)
  margin    - inside ::: {.column-margin} (sidenote/margin prose)
  footnote  - a footnote definition line  [^id]: ...

Non-callout wrapper divs (.content-visible, layout-*, tbl-colwidths, .column-margin)
are TRANSPARENT: we keep auditing inside them (margin gets a zone tag). Only .callout-* is cut.

Output: JSONL, one span per line, with line number + the full source line for context.
The script FLAGS candidates; classification (P1-P4 / AP1-AP4, 7 italic patterns) is a
separate reading pass with full context. It never edits.
"""
import sys, re, json

CALLOUT_RE   = re.compile(r'\.callout-[a-z0-9-]+', re.I)
MARGIN_RE    = re.compile(r'\.column-margin\b', re.I)
DIV_OPEN_RE  = re.compile(r'^\s*:{3,}\s*(\{.*\}|\S.*)?\s*$')   # ':::' or '::::' with content after
DIV_CLOSE_RE = re.compile(r'^\s*:{3,}\s*$')                    # bare colon row -> closes innermost
CODE_FENCE_RE= re.compile(r'^\s*(`{3,}|~{3,})')
HEADING_RE   = re.compile(r'^\s*#{1,6}\s')
FOOTNOTE_RE  = re.compile(r'^\[\^[^\]]+\]:')
ATTR_CAP_RE  = re.compile(r'(fig-cap=|tbl-cap=|lst-cap=|fig-alt=|tbl-alt=|title=)')
TABLE_ROW_RE = re.compile(r'^\s*\|.*\|\s*$')

BOLD_RE   = re.compile(r'\*\*(?P<t>[^*\n]+?)\*\*')
ITALIC_RE = re.compile(r'(?<![*\w])\*(?P<t>[^*\n]+?)\*(?![*\w])')

def mask(line):
    """Hide inline code and math so their * and ** are not read as emphasis."""
    line = re.sub(r'`[^`]*`', lambda m: ' ' * len(m.group()), line)   # inline code
    line = re.sub(r'\$\$.*?\$\$', lambda m: ' ' * len(m.group()), line)
    line = re.sub(r'\$[^$]*\$',  lambda m: ' ' * len(m.group()), line)
    return line

def extract(path):
    out = []
    in_yaml = False
    in_footnote = False        # inside a footnote definition block (dropped)
    code_fence = None          # the fence string that opened the current code block
    div_stack = []             # list of dicts: {callout: bool, margin: bool}
    with open(path, encoding='utf-8') as fh:
        lines = fh.readlines()
    for i, raw in enumerate(lines, 1):
        line = raw.rstrip('\n')

        # YAML frontmatter (only at very top)
        if i == 1 and line.strip() == '---':
            in_yaml = True; continue
        if in_yaml:
            if line.strip() == '---': in_yaml = False
            continue

        # fenced code blocks
        m = CODE_FENCE_RE.match(line)
        if m:
            tok = m.group(1)[0] * 3
            if code_fence is None: code_fence = tok; continue
            elif tok == code_fence: code_fence = None; continue
        if code_fence is not None:
            continue

        # fenced divs (track nesting; cut callout interiors)
        if DIV_CLOSE_RE.match(line):
            if div_stack: div_stack.pop()
            continue
        if DIV_OPEN_RE.match(line):
            attrs = line
            div_stack.append({
                'callout': bool(CALLOUT_RE.search(attrs)),
                'margin':  bool(MARGIN_RE.search(attrs)),
            })
            continue
        # BODY PROSE ONLY: drop callout interiors, margin columns, footnote blocks
        if any(d['callout'] for d in div_stack):   continue   # callouts -> separate pass
        if any(d['margin']  for d in div_stack):   continue   # margin/sidenotes dropped
        if FOOTNOTE_RE.match(line):                            # footnote def -> separate pass
            in_footnote = True; continue
        if in_footnote:
            if line.strip() == '':                  continue  # blank may be mid-footnote
            if re.match(r'^(\s{4,}|\t)', line):     continue  # indented continuation
            in_footnote = False                               # unindented -> footnote ended

        # non-prose single lines
        if HEADING_RE.match(line):       continue
        if TABLE_ROW_RE.match(line):     continue
        if ATTR_CAP_RE.search(line):     continue
        if line.strip().startswith('<!--') or line.strip().startswith('-->'): continue

        zone = 'body'
        masked = mask(line)
        bolds = {(m.start(), m.end()) for m in BOLD_RE.finditer(masked)}
        for mm in BOLD_RE.finditer(masked):
            out.append({'line': i, 'type': 'bold', 'zone': zone,
                        'span': line[mm.start():mm.end()], 'context': line.strip()})
        # italics: drop matches that fall inside a bold span
        for mm in ITALIC_RE.finditer(masked):
            if any(b0 <= mm.start() and mm.end() <= b1 for b0, b1 in bolds): continue
            out.append({'line': i, 'type': 'italic', 'zone': zone,
                        'span': line[mm.start():mm.end()], 'context': line.strip()})
    return out

if __name__ == '__main__':
    spans = extract(sys.argv[1])
    if '--summary' in sys.argv:
        from collections import Counter
        c = Counter((s['type'], s['zone']) for s in spans)
        print(f"file: {sys.argv[1]}")
        print(f"total spans: {len(spans)}")
        for (t, z), n in sorted(c.items()):
            print(f"  {t:6} / {z:8}: {n}")
    else:
        for s in spans:
            print(json.dumps(s, ensure_ascii=False))
