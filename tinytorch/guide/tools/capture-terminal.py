#!/usr/bin/env python3
"""
Capture a real terminal run as an SVG figure for the site.

Every terminal image on the site must come from running the command, not from
a mock-up. This script runs the command with color forced on, keeps its exact
output, and renders it with Rich's SVG exporter, headed by the command itself.

    python3 tools/capture-terminal.py --cwd ~/tinytorch \
        --out assets/images/screenshots/milestone_01_run.svg \
        --lines 1-80 -- tito milestone run perceptron

--lines keeps ranges of output lines (1-based, inclusive, comma-separated, such
as 1-17,128-187) when the full run is too long for one figure. Kept lines are
unedited, and a dim marker shows where lines were left out. Say in the page's
caption what was run and, if --lines was used, that the figure is an excerpt.
"""
import argparse
import os
import shlex
import subprocess
import sys

from rich.console import Console
from rich.text import Text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cwd", default=".", help="directory to run the command in")
    ap.add_argument("--out", required=True, help="SVG file to write")
    ap.add_argument("--width", type=int, default=100, help="terminal width in columns")
    ap.add_argument("--lines", help="keep only these output lines, e.g. 1-17,128-187")
    ap.add_argument("--no-markers", action="store_true", help="do not insert omitted line markers")
    ap.add_argument("--strip-banner", action="store_true", help="strip top TinyTorch banner box from output")
    ap.add_argument("--title", help="window title (default: the command)")
    ap.add_argument("--prompt", help="prompt command to display after $ (default: the executed command)")
    ap.add_argument("command", nargs=argparse.REMAINDER, help="command to run, after --")
    args = ap.parse_args()

    cmd = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not cmd:
        ap.error("give the command after --")

    env = dict(os.environ, FORCE_COLOR="1", TERM="xterm-256color", COLUMNS=str(args.width))
    proc = subprocess.run(cmd, cwd=os.path.expanduser(args.cwd), env=env, stdin=subprocess.DEVNULL,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = proc.stdout.decode("utf-8", errors="replace").replace("\r\n", "\n")
    all_lines = out.split("\n")
    while all_lines and not Text.from_ansi(all_lines[-1]).plain.strip():
        all_lines.pop()

    if args.strip_banner:
        # Check if first lines contain the TinyTorch banner box (surrounded by ╭...╮ or ┌...┐)
        first_text = "\n".join(Text.from_ansi(l).plain for l in all_lines[:15])
        if "Don't import it. Build it" in first_text or "Tiny" in first_text and "TORCH" in first_text:
            # Find the closing line of the top banner
            for idx, l in enumerate(all_lines[:15]):
                plain = Text.from_ansi(l).plain.strip()
                if plain.startswith("╰") or plain.startswith("└") or (idx > 0 and plain == ""):
                    all_lines = all_lines[idx + 1:]
                    # Drop leading empty lines
                    while all_lines and not Text.from_ansi(all_lines[0]).plain.strip():
                        all_lines.pop(0)
                    break

    pieces = []  # (is_marker, text)
    if args.lines:
        prev_last = 0
        for rng in args.lines.split(","):
            first, last = (int(x) for x in rng.split("-"))
            if not args.no_markers and first > prev_last + 1:
                pieces.append((True, f"  ⋮  output lines {prev_last + 1}-{first - 1} omitted"))
            pieces.extend((False, l) for l in all_lines[first - 1:last])
            prev_last = last
        if not args.no_markers and prev_last < len(all_lines):
            pieces.append((True, f"  ⋮  output lines {prev_last + 1}-{len(all_lines)} omitted"))
    else:
        pieces = [(False, l) for l in all_lines]

    # Sanitize machine-local paths like /Users/... to ~/
    home = os.path.expanduser("~")
    sanitized_pieces = []
    for is_marker, line in pieces:
        if not is_marker and home in line:
            line = line.replace(home, "~")
        sanitized_pieces.append((is_marker, line))

    shown = args.prompt or " ".join(shlex.quote(c) for c in cmd)
    console = Console(record=True, width=args.width, force_terminal=True, color_system="truecolor",
                      file=open(os.devnull, "w"))
    console.print(Text("$ ", style="bold green") + Text(shown, style="bold"), soft_wrap=True)
    for is_marker, line in sanitized_pieces:
        if is_marker:
            console.print()
            console.print(Text(line, style="dim italic"), soft_wrap=True)
            console.print()
        else:
            console.print(Text.from_ansi(line), soft_wrap=True)
    console.save_svg(args.out, title=args.title or shown)
    # Ensure generated SVG contains valid XML characters and no machine-local paths
    import re
    with open(args.out, "rb") as f:
        raw_svg = f.read()
    clean_svg = re.sub(b'[\x00-\x08\x0b\x0c\x0e-\x1f]', b'', raw_svg)
    clean_svg = re.sub(rb'/Users/[a-zA-Z0-9_.-]+', b'~', clean_svg)
    if clean_svg != raw_svg:
        with open(args.out, "wb") as f:
            f.write(clean_svg)
    kept = sum(1 for m, _ in pieces if not m)
    print(f"wrote {args.out}: {kept} of {len(all_lines)} output lines, command exit status {proc.returncode}")
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
