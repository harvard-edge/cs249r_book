#!/usr/bin/env python3
"""Emit the D1 schema DDL from the compiler module.

Output file lives at ``interviews/vault-cli/scripts/d1-schema.sql`` — committed
so ``wrangler d1 execute ... --file`` can apply it to a fresh D1 instance.

The schema fingerprint in wrangler.toml should be set to SHA-256 of the
normalized DDL (whitespace-collapsed) so the Worker's cold-start check can
verify the D1 instance matches what was published.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vault_cli.compiler import DDL  # noqa: E402


def main() -> int:
    out = Path(__file__).parent / "d1-schema.sql"
    out.write_text(DDL.strip() + "\n", encoding="utf-8")

    normalized = re.sub(r"\s+", " ", DDL).strip()
    fingerprint = hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    print(f"wrote {out}")
    print(f"schema_fingerprint: {fingerprint}")
    print(f"  set SCHEMA_FINGERPRINT={fingerprint} in wrangler.toml after each DDL change.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
