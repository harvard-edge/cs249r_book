# Archived Lua filters

Lua filters that were authored at some point but are not currently wired into
any active filter chain. Kept here so the work is not lost, but excluded from
the live build.

| File | Purpose | Status |
|------|---------|--------|
| `inject_glossary.lua` | Wraps glossary terms in tooltip / margin / link markup based on output format. | Not referenced in any `filters.yml`. Reads `data/master_glossary.json` (path no longer exists at the expected location). |
| `auto-glossary.lua` | Earlier auto-glossary detector. | Superseded by `inject_glossary.lua`. |
| `auto-glossary-advanced.lua` | Most ambitious variant of the auto-glossary detector. | Never wired up. |

To revive any of these: move the file back to `book/quarto/filters/`, add it to
the appropriate `book/quarto/config/shared/{html,pdf,epub}/filters.yml`, and
verify the `dkjson` Lua dependency and glossary JSON paths still resolve.

The glossary is currently rendered statically from `book/quarto/contents/<vol>/backmatter/glossary/` rather than via a Lua filter pass.
