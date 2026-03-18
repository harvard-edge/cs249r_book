# Labs Test Runner
# Usage: cd labs && make test

.PHONY: test test-static test-engine test-widget test-wasm help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

test: ## Run all test levels (static + engine + widget)
	python3 -m pytest tests/ -v

test-static: ## Level 1: Fast static analysis (syntax, structure, imports)
	python3 -m pytest tests/test_static.py -v

test-engine: ## Level 2: Headless cell execution via app.run()
	python3 -m pytest tests/test_engine.py -v -k engine

test-widget: ## Level 3: Widget structure and prediction-reveal flow
	python3 -m pytest tests/test_widget.py -v -k widget

test-wasm: ## Smoke test: export 3 representative labs to WASM HTML
	@echo "Smoke-testing WASM export for 3 representative labs..."
	@mkdir -p /tmp/wasm-smoke
	@PASS=0; FAIL=0; \
	for lab in vol1/lab_00_introduction.py vol1/lab_01_ml_intro.py vol2/lab_01_introduction.py; do \
		name=$$(basename "$$lab" .py); \
		if marimo export html-wasm "$$lab" -o "/tmp/wasm-smoke/$$name/index.html" --mode run --no-show-code 2>/dev/null; then \
			SIZE=$$(wc -c < "/tmp/wasm-smoke/$$name/index.html"); \
			if [ "$$SIZE" -gt 10000 ]; then \
				echo "  OK: $$name ($$SIZE bytes)"; PASS=$$((PASS + 1)); \
			else \
				echo "  FAIL: $$name (only $$SIZE bytes)"; FAIL=$$((FAIL + 1)); \
			fi; \
		else \
			echo "  FAIL: $$name (export error)"; FAIL=$$((FAIL + 1)); \
		fi; \
	done; \
	echo ""; \
	echo "Results: $$PASS passed, $$FAIL failed"; \
	[ "$$FAIL" -eq 0 ] || exit 1
