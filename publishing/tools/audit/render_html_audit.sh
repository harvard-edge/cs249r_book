#!/usr/bin/env bash
# Back-compat wrapper — prefer book/tools/audit/fmt/render_html.sh
exec "$(dirname "$0")/fmt/render_html.sh" "$@"
