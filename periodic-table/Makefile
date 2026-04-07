# Periodic Table of ML Systems — build orchestration.
#
# table.yml is the single source of truth. Both index.html (the standalone
# shareable artifact) and the StaffML React data file are generated from it.
#
# Usage:
#   make           — same as `make all`
#   make all       — validate the YAML, regenerate index.html, sync the React TS file
#   make html      — regenerate periodic-table/index.html from table.yml
#   make sync      — regenerate interviews/staffml/src/data/periodicTable.ts from table.yml
#   make validate  — schema-check table.yml + run cross-reference invariants
#   make clean     — remove generated files (DESTRUCTIVE — only run after committing)

REPO := $(shell cd .. && pwd)
YAML := table.yml
HTML := index.html
TS   := $(REPO)/interviews/staffml/src/data/periodicTable.ts

NODE := node

.PHONY: all html sync validate clean help

all: html sync
	@echo "✓ Built $(HTML) and $(TS) from $(YAML)"

html:
	@$(NODE) scripts/build-html.mjs

sync:
	@$(NODE) $(REPO)/interviews/staffml/scripts/sync-periodic-table.mjs

validate:
	@$(NODE) scripts/validate.mjs

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'
