#!/usr/bin/env bash
# One-time setup: point git to hooks that run pre-commit on all files (matches CI).
set -e
cd "$(git rev-parse --show-toplevel)"
git config core.hooksPath book/tools/git-hooks
echo "✅ Git hooks configured. Commits will run pre-commit on all files (same as CI)."
