#!/usr/bin/env python3
"""Verify every `tito ...` command shown in the guide actually exists.

The lab guide is the CLI's public surface: if a page tells a student to run
`tito module complete 03 --verbose`, that has to work. It did not -- --verbose
is a global flag, so the documented form exited 2 with "unrecognized
arguments". Nothing caught it, because prose and CLI drift independently.

This walks tito's real argparse tree, extracts every `tito ...` invocation from
the .qmd sources, and fails on an unknown subcommand, an unknown flag, or an
invalid value for a flag that declares choices.

Usage:  python3 guide/tools/check-documented-commands.py [guide_dir]
Exit:   0 clean, 1 if any documented command would not run.
"""
import argparse
import collections
import os
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
TINYTORCH = HERE.parent.parent
sys.path.insert(0, str(TINYTORCH))


def _reexec_under_project_env() -> None:
    """Re-run under the project interpreter if this one lacks tito's deps.

    Importing tito pulls in numpy, which pre-commit's `language: system`
    python does not have. Rather than let the hook die on ModuleNotFoundError
    (or, worse, pass vacuously), hand off to the venv interpreter once.
    """
    if os.environ.get("_TITO_DOC_CHECK_REEXEC"):
        return
    for candidate in (TINYTORCH / ".venv/bin/python3",
                      TINYTORCH.parent / ".venv/bin/python3"):
        if candidate.exists():
            os.environ["_TITO_DOC_CHECK_REEXEC"] = "1"
            os.execv(str(candidate), [str(candidate), __file__, *sys.argv[1:]])

# A token standing in for a positional the reader fills in (01, N, <name>, ...).
PLACEHOLDER = re.compile(r"^(<.+>|\{.+\}|N|NN|\d+|XX|[A-Z_]{2,})$")
# `tito` appearing as an English noun rather than as a command.
PROSE = re.compile(r"^(command|commands|folder|directory|CLI|itself|stages|reads)$")

INVOCATION = re.compile(r"\btito\b((?:[ \t]+[^\s`\"'<>|;&)\]]+)*)")


def cli_tree():
    try:
        from tito.main import TinyTorchCLI
    except ImportError:
        _reexec_under_project_env()          # does not return if it can hand off
        raise
    paths, flags, choices = set(), {}, {}

    def walk(parser, prefix):
        paths.add(prefix)
        opts = set()
        for a in parser._actions:
            if isinstance(a, argparse._SubParsersAction):
                for name, sub in a.choices.items():
                    walk(sub, f"{prefix} {name}")
            else:
                opts.update(a.option_strings)
                if a.choices and a.option_strings:
                    choices[(prefix, a.option_strings[0])] = {str(c) for c in a.choices}
        flags[prefix] = opts

    walk(TinyTorchCLI().create_parser(), "tito")
    return paths, flags, choices


def main() -> int:
    root = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent
    try:
        paths, flags, choices = cli_tree()
    except ImportError as exc:
        # No project environment anywhere (fresh clone, no venv). Say so loudly
        # rather than reporting a clean run nobody actually performed.
        msg = (f"cannot import tito ({exc}). Run `tito setup`, or "
               f"`pip install -e tinytorch`, to enable this check.")
        if os.environ.get("CI"):
            # CI always has the environment, so a skip there means the gate has
            # silently stopped running. Fail instead of reporting a clean run.
            print(f"FAILED: {msg}")
            return 1
        print(f"SKIPPED: {msg}")
        return 0
    problems = collections.defaultdict(list)

    for f in sorted(root.rglob("*.qmd")):
        for lineno, line in enumerate(f.read_text(encoding="utf-8",
                                                  errors="replace").splitlines(), 1):
            for m in INVOCATION.finditer(line):
                toks = m.group(1).split()
                where = f"{f.relative_to(root)}:{lineno}"
                cur, i = "tito", 0
                while i < len(toks) and not toks[i].startswith("-"):
                    # Prose runs a command into a clause: "...with tito module
                    # complete, and inspect...". Strip the punctuation before
                    # deciding the token is not a subcommand.
                    tok = toks[i].rstrip(".,;:)")
                    nxt = f"{cur} {tok}"
                    if nxt in paths:
                        cur, i = nxt, i + 1
                        continue
                    has_children = any(
                        p.startswith(cur + " ") and p.count(" ") == cur.count(" ") + 1
                        for p in paths
                    )
                    if has_children and not PLACEHOLDER.match(tok) and not PROSE.match(tok):
                        problems[f"unknown subcommand: `{cur} {tok}`"].append(where)
                    break

                known = flags.get(cur, set())
                rest = toks[i:]
                for j, tok in enumerate(rest):
                    if not tok.startswith("--"):
                        continue
                    name, _, inline = tok.partition("=")
                    name = name.rstrip(".,;:)")
                    if name not in known:
                        problems[f"unknown flag: `{name}` on `{cur}`"].append(where)
                        continue
                    allowed = choices.get((cur, name))
                    if allowed:
                        val = inline or (rest[j + 1] if j + 1 < len(rest) else "")
                        val = val.rstrip(".,;:)")
                        if val and not val.startswith("-") and val not in allowed:
                            problems[
                                f"invalid value `{val}` for `{name}` on `{cur}` "
                                f"(expected one of {sorted(allowed)})"
                            ].append(where)

    if not problems:
        print("Documented tito commands: all valid.")
        return 0

    print("Documented tito commands that would not run:\n")
    for issue in sorted(problems):
        print(f"  {issue}")
        for loc in sorted(set(problems[issue]))[:8]:
            print(f"      {loc}")
        print()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
