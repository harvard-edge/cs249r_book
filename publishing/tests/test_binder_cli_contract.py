from pathlib import Path

from book.cli.checks.cli_contract import CASES, run_contract


ROOT = Path(__file__).resolve().parents[2]


def test_binder_cli_contract_is_stable():
    violations = run_contract(ROOT)

    rendered = "\n".join(
        f"{v.code}: {v.message}\n{v.context}\n{v.suggestion}"
        for v in violations
    )
    assert not violations, rendered
    assert len(CASES) >= 10
