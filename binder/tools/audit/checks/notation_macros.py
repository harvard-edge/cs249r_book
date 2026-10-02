from pathlib import Path
import re
import yaml
from audit.ledger import Issue, make_issue_id

CATEGORY = "notation-macros"
RULE = "All defined notations must use their respective LaTeX macros"
RULE_TEXT = (
    "Raw mathematical notation variables defined in notation_registry.yml "
    "MUST be referenced using their semantic \\macro equivalent. "
    "Do not use the raw string; use the macro to prevent drift. "
    "If you are an LLM agent, read books/shared/_includes/_macros.tex to see all available macros."
)

import os
# Load registry once at module level
registry_path = Path(os.environ.get("BOOK_ROOT", ".")) / "books" / "shared" / "_includes" / "notation_registry.yml"
if not registry_path.exists():
    # fallback for local testing
    registry_path = Path("books/shared/_includes/notation_registry.yml")

banned_raw = {}
if registry_path.exists():
    with open(registry_path) as f:
        registry = yaml.safe_load(f)
    for var in registry["variables"]:
        latex = var["latex"].strip()
        macro = var["macro"].strip()
        if "\\text{" in latex or "_" in latex:
            banned_raw[latex] = macro

def check(file_path: Path, text: str, lines: list[str]) -> list[Issue]:
    issues = []
    
    # Skip notation guide files themselves
    if "_includes" in str(file_path) or "frontmatter" in str(file_path):
        return issues
        
    for i, line in enumerate(lines):
        # Only check lines that might contain math
        if "$" not in line and "\\[" not in line and "\\(" not in line:
            continue
            
        for latex, macro in banned_raw.items():
            # simple string search for the exact raw latex
            if latex in line:
                issues.append(
                    {
                        "id": make_issue_id(CATEGORY, file_path, i + 1, "raw_notation"),
                        "file": str(file_path),
                        "line": i + 1,
                        "category": CATEGORY,
                        "rule": RULE,
                        "message": f"Found raw notation `{latex}`. Use the semantic macro `{macro}` instead.\n{RULE_TEXT}",
                        "status": "OPEN",
                        "severity": "error",
                    }
                )
    return issues
