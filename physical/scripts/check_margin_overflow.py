#!/usr/bin/env python3
"""
DEPRECATED: Use './psyai layout' or './pai layout' instead.
This script forwards all arguments to the unified Physical AI CLI.
"""
import sys
import subprocess
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
cmd = [sys.executable, str(repo_root / "cli.py"), "layout"] + sys.argv[1:]
sys.exit(subprocess.run(cmd).returncode)
