#!/usr/bin/env python3
"""Enforces strict bibliography boundaries between project components."""

import re
import sys
import argparse
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parents[3]

# Define strict scopes: {Name: {"sources": [dirs/files], "bibs": [files]}}
SCOPES = {
    "Vol1": {
        "sources": ["book/quarto/contents/vol1"],
        "bibs": ["book/quarto/contents/vol1/backmatter/references.bib"]
    },
    "Vol2": {
        "sources": ["book/quarto/contents/vol2"],
        "bibs": ["book/quarto/contents/vol2/backmatter/references.bib"]
    },
    "Interviews": {
        "sources": ["interviews"],
        "bibs": ["interviews/paper/references.bib"]
    },
    "TinyTorch": {
        "sources": ["tinytorch"],
        "bibs": ["tinytorch/paper/references.bib"]
    },
    "MLSysIm": {
        "sources": ["mlsysim"],
        "bibs": [
            "mlsysim/paper/references.bib",
            "mlsysim/docs/references.bib"
        ]
    },
    "PeriodicTable": {
        "sources": ["periodic-table"],
        "bibs": ["periodic-table/paper/references.bib"]
    }
}

EXCLUDE = ("_build", "_site", "node_modules", ".git", "__pycache__", ".venv")
NON_CITE_PREFIXES = ("sec-", "fig-", "tbl-", "eq-", "lst-", "exr-", "exm-", "thm-")

# Tightened regex: Must NOT be followed by a / (which suggests a file path)
# and should be followed by a word boundary or specific punctuation.
CITE_RE = re.compile(r"(?<![=,(])\[?@([A-Za-z][\w:.-]*)(?![/])\b")
TEX_CITE_RE = re.compile(r"\\cite[a-z]*\*?\{([^}]+)\}")

def parse_bib_keys(path):
    if not path.exists(): return set()
    text = path.read_text(encoding="utf-8")
    return set(re.findall(r"^@\w+\s*\{\s*([\w:_-]+)\s*,", text, re.M))

# Keys that look like CSS / JS / Python decorators / emails — false positives
KNOWN_FALSE_POSITIVE_KEYS = {
    "media", "keyframes", "import", "supports", "page", "font-face",
    "charset", "namespace", "document",  # CSS at-rules
    "grad", "staticmethod", "classmethod", "property", "abstractmethod",
    "dataclass", "cached_property", "wraps",  # Python decorators
    "eecs.harvard.edu", "harvard.edu", "google.com",  # emails
    "B", "B.T", "W", "X", "Y", "A", "C", "Z", # Math variables
}

def extract_cites(path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix == ".tex":
        text = re.sub(r"(?<!\\)%.*", "", text)
        keys = set()
        for m in TEX_CITE_RE.finditer(text):
            for k in m.group(1).split(","):
                k = k.strip()
                if k and k not in KNOWN_FALSE_POSITIVE_KEYS: 
                    keys.add(k)
        return keys
    else: # .qmd
        text = re.sub(r"```.*?```", "", text, flags=re.S)
        text = re.sub(r"`[^`]+`", "", text)
        keys = set()
        for m in CITE_RE.finditer(text):
            k = m.group(1).rstrip(".,;:)")
            if k and not k.startswith(NON_CITE_PREFIXES) and k not in KNOWN_FALSE_POSITIVE_KEYS:
                keys.add(k)
        return keys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    all_defined_keys = {} # key -> list of bibs
    scope_bib_keys = {}   # scope -> set of keys

    # 1. Load all bibliographies
    for scope, cfg in SCOPES.items():
        keys = set()
        for bp in cfg["bibs"]:
            path = REPO_ROOT / bp
            pkeys = parse_bib_keys(path)
            keys.update(pkeys)
            for k in pkeys:
                if k not in all_defined_keys: all_defined_keys[k] = []
                all_defined_keys[k].append(bp)
        scope_bib_keys[scope] = keys

    # 2. Scan sources per scope
    exit_code = 0
    print(f"Checking {len(SCOPES)} citation boundaries...\n")

    for scope, cfg in SCOPES.items():
        found_cites = defaultdict(list) # key -> list of files
        for src_root in cfg["sources"]:
            p = REPO_ROOT / src_root
            if not p.exists(): continue
            files = list(p.rglob("*.qmd")) + list(p.rglob("*.tex"))
            for f in files:
                if any(part in EXCLUDE for part in f.parts): continue
                cites = extract_cites(f)
                rel = str(f.relative_to(REPO_ROOT))
                for k in cites:
                    found_cites[k].append(rel)

        cited_keys = set(found_cites.keys())
        local_bib = scope_bib_keys[scope]
        
        unresolved = sorted(cited_keys - local_bib)
        orphans = sorted(local_bib - cited_keys)

        print(f"=== Scope: {scope} ===")
        print(f"  Bib(s):   {', '.join(cfg['bibs'])}")
        print(f"  Citations: {len(cited_keys)} unique keys")
        
        if not unresolved and not orphans:
            print("  ✅ Integrity perfect.")
        else:
            if unresolved:
                print(f"  ❌ {len(unresolved)} unresolved or boundary-leaking citations:")
                for k in unresolved:
                    other_bibs = all_defined_keys.get(k)
                    if other_bibs:
                        print(f"     - {k} (LEAK: Found in {', '.join(other_bibs)})")
                    else:
                        print(f"     - {k} (MISSING: Not found in any bib)")
                    if args.verbose:
                        print(f"       Cited in: {', '.join(found_cites[k][:3])}")
                exit_code = 1
            
            if orphans:
                print(f"  ⚠️ {len(orphans)} scope-specific orphans (defined but not cited in this scope)")
                if args.verbose:
                    for k in orphans: print(f"     - {k}")
        print()

    return exit_code

if __name__ == "__main__":
    sys.exit(main())
