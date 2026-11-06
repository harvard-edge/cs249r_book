"""Production pruning keeps staging-only Labs destinations out of the portal."""
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / 'labs/index.qmd'
STRIP = ROOT / '.github/scripts/strip-unreleased.py'


def test_live_strip_removes_hardware_studio_link_but_dev_source_keeps_it(tmp_path):
    source = INDEX.read_text()
    assert '[Volume IV Physical AI Hardware Studio](vol4/README.md)' in source
    assert '<summary>Local notebooks and hardware</summary>' in source

    staged_index = tmp_path / 'index.qmd'
    staged_index.write_text(source)
    result = subprocess.run(
        [sys.executable, str(STRIP), str(staged_index)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    published = staged_index.read_text()
    assert '<summary>Local notebooks and hardware</summary>' in published
    assert 'vol4/README.md' not in published
    assert 'Volume IV Physical AI Hardware Studio' not in published
    assert INDEX.read_text() == source  # Dev preview skips this production-only transform.
