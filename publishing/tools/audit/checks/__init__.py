"""Audit check functions, one module per category.

Each check module exports a single function with the signature:

    def check(file_path: Path, text: str, lines: list[str]) -> list[Issue]

where Issue is the dict format defined in book/tools/audit/ledger.py.
The scanner imports all check modules in this package and dispatches.
"""
