#!/usr/bin/env python3
"""
Extract code listings for the narrative book from the TinyTorch source modules.

The book quotes the code by symbol name. `listings.yml` (next to this script)
declares every listing:

    - target: _listings/06_autograd/Function.qmd
      module: 01_tensor
      symbol: Function              # a class, a function, Class.method, or a
                                    # method attached with @method_of(Class)
      elide: [__repr__]             # optional: methods whose bodies become `...`
      keep_scaffold: false          # optional: keep TODO/APPROACH/HINT text

Running the script regenerates every target as a fenced ```python block that a
chapter pulls in with `{{< include _listings/06_autograd/Function.qmd >}}`.
Student scaffolding (TODO, APPROACH, EXAMPLE, HINT blocks and the nbgrader
solution markers) is stripped so the listing reads as framework code.

    python3 tools/listings.py            # regenerate all targets
    python3 tools/listings.py --check    # exit 1 if any target is stale or a symbol is missing

A symbol that cannot be found is an error, so renaming a function in `src/`
fails the book build until the chapter that quotes it is updated.
"""
import argparse
import ast
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BOOK = HERE.parent
SRC = BOOK.parent / "src"
MANIFEST = HERE / "listings.yml"

SCAFFOLD_HEAD = re.compile(
    r"^\s*(TODO|APPROACH|EXAMPLE|HINTS?|LOOP STRUCTURE|MATHEMATICAL FORMULA|"
    r"WEIGHT INITIALIZATION|IMPLEMENTATION|STEPS?)\b"
)


def module_source(module: str) -> str:
    path = SRC / module / f"{module}.py"
    if not path.exists():
        raise FileNotFoundError(f"no module source at {path}")
    text = path.read_text()
    # Drop cell headers and nbdev directives; they are not Python.
    lines = [l for l in text.split("\n") if not l.startswith("# %%") and not l.startswith("#|")]
    return "\n".join(lines)


def find_node(tree: ast.Module, symbol: str):
    """Locate a top-level def/class, a Class.method, or a method attached with @method_of(Class)."""
    if "." in symbol:
        cls_name, meth = symbol.split(".", 1)
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == cls_name:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == meth:
                        return item
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == meth:
                for dec in node.decorator_list:
                    if (isinstance(dec, ast.Call) and getattr(dec.func, "id", None) == "method_of"
                            and dec.args and getattr(dec.args[0], "id", None) == cls_name):
                        return node
        raise KeyError(symbol)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == symbol:
            return node
    raise KeyError(symbol)


def node_text(lines, node) -> str:
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])]) - 1
    return "\n".join(lines[start: node.end_lineno])


def strip_docstring_scaffold(code: str) -> str:
    """Inside every docstring, drop everything from the first TODO/APPROACH/HINT line on."""
    out, in_doc, dropping, indent = [], False, False, ""
    for line in code.split("\n"):
        stripped = line.strip()
        if not in_doc:
            out.append(line)
            if stripped.count('"""') == 1 and (stripped.startswith('"""') or stripped.startswith('r"""')):
                in_doc, dropping = True, False
                indent = line[: len(line) - len(line.lstrip())]
            continue
        # inside a docstring
        if '"""' in stripped:
            # closing line: trim trailing blank lines we kept, then close
            while out and out[-1].strip() == "" and not out[-1].endswith('"""'):
                out.pop()
            if stripped == '"""':
                out.append(indent + '"""')
            else:
                out.append(line if not dropping else indent + '"""')
            in_doc = False
            continue
        if dropping:
            continue
        if SCAFFOLD_HEAD.match(line):
            dropping = True
            continue
        out.append(line)
    return "\n".join(out)


def first_paragraph_docstrings(code: str) -> str:
    """Keep only the first paragraph of every docstring (the book quotes code, not homework)."""
    out, in_doc, cutting, indent = [], False, False, ""
    for line in code.split("\n"):
        stripped = line.strip()
        if not in_doc:
            out.append(line)
            if stripped.count('"""') == 1 and stripped.startswith('"""'):
                in_doc, cutting = True, False
                indent = line[: len(line) - len(line.lstrip())]
                if stripped == '"""':
                    pass  # opening line alone; first paragraph follows
            continue
        if '"""' in stripped:
            if cutting:
                out.append(indent + '"""')
            else:
                out.append(line)
            in_doc = False
            continue
        if cutting:
            continue
        if stripped == "":
            cutting = True
            continue
        out.append(line)
    return "\n".join(out)


def strip_markers(code: str) -> str:
    return "\n".join(l for l in code.split("\n") if "### BEGIN SOLUTION" not in l and "### END SOLUTION" not in l)


def elide_methods(code: str, names) -> str:
    if not names:
        return code
    lines = code.split("\n")
    tree = ast.parse(code)
    spans = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            first_body = node.body[0]
            body_start = first_body.lineno - 1
            if isinstance(first_body, ast.Expr) and isinstance(getattr(first_body, "value", None), ast.Constant) and isinstance(first_body.value.value, str):
                body_start = first_body.end_lineno  # keep the docstring, elide the rest
            spans.append((body_start, node.end_lineno, node.col_offset + 4))
    for start, end, col in sorted(spans, reverse=True):
        lines[start:end] = [" " * col + "..."]
    return "\n".join(lines)


def collapse_blank_runs(code: str) -> str:
    code = "\n".join(l.rstrip() for l in code.split("\n"))   # no trailing whitespace (pre-commit strips it)
    return re.sub(r"\n{3,}", "\n\n", code).strip("\n") + "\n"


def extract(module: str, symbol: str, elide=(), keep_scaffold=False, doc="first") -> str:
    text = module_source(module)
    tree = ast.parse(text)
    node = find_node(tree, symbol)
    code = node_text(text.split("\n"), node)
    code = strip_markers(code)
    if not keep_scaffold:
        code = strip_docstring_scaffold(code)
    if doc == "first":
        code = first_paragraph_docstrings(code)
    code = elide_methods(code, set(elide))
    # A method quoted on its own is shown at column zero.
    if "." in symbol and code.startswith("    "):
        code = "\n".join(l[4:] if l.startswith("    ") else l for l in code.split("\n"))
    return collapse_blank_runs(code)


def load_manifest():
    import yaml  # PyYAML ships with the book toolchain
    entries = yaml.safe_load(MANIFEST.read_text()) or []
    for e in entries:
        for key in ("target", "module", "symbol"):
            if key not in e:
                raise ValueError(f"manifest entry missing {key}: {e}")
    return entries


def render(entry) -> str:
    code = extract(entry["module"], entry["symbol"], entry.get("elide", ()), entry.get("keep_scaffold", False), entry.get("doc", "first"))
    header = f"<!-- generated by tools/listings.py from src/{entry['module']}/{entry['module']}.py :: {entry['symbol']} -->\n"
    return header + "```python\n" + code + "```\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify targets are up to date; do not write")
    ap.add_argument("--show", nargs=2, metavar=("MODULE", "SYMBOL"), help="print one extracted listing and exit")
    args = ap.parse_args(argv)
    if args.show:
        sys.stdout.write(extract(*args.show))
        return 0
    stale, missing = [], []
    for entry in load_manifest():
        target = BOOK / entry["target"]
        try:
            text = render(entry)
        except (KeyError, FileNotFoundError) as e:
            missing.append(f"{entry['target']}: {entry['module']}::{entry['symbol']} ({e})")
            continue
        if args.check:
            if not target.exists() or target.read_text() != text:
                stale.append(entry["target"])
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or target.read_text() != text:
                target.write_text(text)
                print("wrote", entry["target"])
    if missing:
        print("MISSING SYMBOLS (rename in src/ without updating the chapter?):", file=sys.stderr)
        for m in missing:
            print("  " + m, file=sys.stderr)
    if stale:
        print("STALE LISTINGS (run tools/listings.py):", file=sys.stderr)
        for s in stale:
            print("  " + s, file=sys.stderr)
    return 1 if (missing or stale) else 0


if __name__ == "__main__":
    sys.exit(main())
