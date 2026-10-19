"""Measure the GitHub MCP server's tool descriptors (one row per tool name).

Input: the __toolsnaps__/*.snap files of github/github-mcp-server at a pinned
commit. Tokens: cl100k_base (tiktoken) over the compact JSON of the
model-visible part, {name, description, input_schema}; provider framing is not
counted. Usage: python3 collect_tools.py SNAPDIR OUT.json
"""
import json, sys
from pathlib import Path
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
seen = {}
for f in sorted(Path(sys.argv[1]).glob("*.snap")):
    d = json.loads(f.read_text())
    if "name" not in d or "inputSchema" not in d:
        continue
    if d["name"] in seen and f.name != d["name"] + ".snap":
        continue
    seen[d["name"]] = d
rows = []
for name, d in sorted(seen.items()):
    view = json.dumps({"name": name, "description": d.get("description", ""),
                       "input_schema": d["inputSchema"]}, separators=(",", ":"))
    a = d.get("annotations", {}); p = d["inputSchema"].get("properties", {})
    rows.append(dict(name=name, tokens=len(enc.encode(view)), chars=len(view),
                     read_only=a.get("readOnlyHint") is True,
                     idempotent=a.get("idempotentHint") is True,
                     destructive=a.get("destructiveHint") is True,
                     repo_free_string=("repo" in p and "enum" not in p["repo"])))
json.dump(rows, open(sys.argv[2], "w"), indent=0)
print(len(rows), "tools")
