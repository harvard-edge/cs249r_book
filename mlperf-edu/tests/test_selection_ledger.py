from __future__ import annotations

from tools.check_selection_ledger import validate


def test_selection_ledger_is_complete():
    assert validate() == []
