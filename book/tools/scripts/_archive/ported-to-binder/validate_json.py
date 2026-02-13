#!/usr/bin/env python3
"""Validate JSON files using Python's built-in json module."""
import json
import sys

exit_code = 0
for filepath in sys.argv[1:]:
    try:
        with open(filepath, 'r') as f:
            json.load(f)
        print(f"✅ {filepath}")
    except json.JSONDecodeError as e:
        print(f"❌ {filepath}: {e}")
        exit_code = 1
    except Exception as e:
        print(f"⚠️  {filepath}: {e}")
        exit_code = 1

sys.exit(exit_code)
