from __future__ import annotations

import yaml

from tools.check_selection_ledger import LEDGER, UniqueKeySafeLoader, validate


def test_selection_ledger_is_complete():
    assert validate() == []


def test_selection_ledger_records_every_portfolio_decision():
    data = yaml.load(LEDGER.read_text(encoding="utf-8"), Loader=UniqueKeySafeLoader)
    statuses = [entry["status"] for entry in data["workloads"].values()]

    assert len(statuses) == 17
    assert statuses.count("candidate") == 14
    assert statuses.count("deferred") == 0
    assert statuses.count("rejected") == 3


def test_selection_ledger_rejects_duplicate_yaml_keys(tmp_path):
    ledger = tmp_path / "selection-ledger.yaml"
    ledger.write_text(
        "schema: mlperf-edu-workload-selection/0.1\n"
        "workloads:\n"
        "  duplicate:\n"
        "    status: deferred\n"
        "    status: rejected\n",
        encoding="utf-8",
    )

    errors = validate(ledger)
    assert len(errors) == 1
    assert "duplicate key 'status'" in errors[0]
