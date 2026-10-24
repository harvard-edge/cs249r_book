import json, re, sys, statistics as st
from pathlib import Path
d = Path(sys.argv[1]); out = Path(sys.argv[2])
rows = []
for f in sorted(d.glob("*.json")):
    t = json.load(open(f))
    for x in t["messages"]:
        if x["role"] != "user": continue
        c = x["content"] if isinstance(x["content"], str) else " ".join(p.get("text","") for p in x["content"])
        m = re.match(r"<returncode>(-?\d+)</returncode>", c)
        if not m: continue
        rc = int(m.group(1))
        e = re.search(r"<elided_chars>\s*(\d+) characters elided", c)
        if e:
            h = re.search(r"<output_head>\n?(.*?)</output_head>", c, re.S).group(1)
            tl = re.search(r"<output_tail>\n?(.*?)</output_tail>", c, re.S)
            tl = tl.group(1) if tl else ""
            raw = len(h) + int(e.group(1)) + len(tl); staged = len(c); trunc = 1
        else:
            o = re.search(r"<output>\n?(.*?)</output>", c, re.S)
            raw = len(o.group(1)) if o else 0; staged = len(c); trunc = 0
        rows.append((f.stem, rc, trunc, raw, staged))
with open(out, "w") as w:
    w.write("instance_id,returncode,truncated,raw_chars,staged_chars\n")
    for r in rows: w.write(",".join(map(str, r)) + "\n")
n = len(rows); tr = [r for r in rows if r[2]]
print("observations", n, "truncated", len(tr), "share", len(tr)/n)
print("runs with a truncation", len({r[0] for r in tr}))
print("max raw", max(r[3] for r in rows), "median raw", st.median(r[3] for r in rows))
print("max staged", max(r[4] for r in rows))
print("raw of truncated: median", st.median(r[3] for r in tr), "max", max(r[3] for r in tr))
print("rc0 share", sum(1 for r in rows if r[1]==0)/n, "rc dist", sorted({r[1] for r in rows}))
