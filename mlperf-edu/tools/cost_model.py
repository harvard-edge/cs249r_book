"""
Iter-9 (Huyen): cost-per-1k-token model.

Two cost numbers per workload, both publicly priced and measurable:

  energy_floor:   $/1k-tok = (J/token × 1000) × ($/kWh × 2.778e-7)
                  -> what the laptop literally consumes.
  cloud_ceiling:  $/1k-tok = ($_per_hour_SKU / 3600) × seconds_per_1k_tok
                  -> what your employer pays for an underutilized rental.

The gap between the two is the utilization story. Capex amortization is
intentionally NOT modeled (varies 5x by assumption); see Huyen's iter-9
proposal for the rationale.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SKUS_YAML = REPO_ROOT / "tools" / "cost_skus.yaml"

# US average industrial electricity rate, 2025 (EIA Table 5.6.A).
DEFAULT_USD_PER_KWH = 0.10

# 1 kWh = 3.6e6 J; convert J -> kWh by dividing.
_J_TO_KWH = 1.0 / 3.6e6


class CostQuote(NamedTuple):
    energy_floor_per_1k_tok: float
    cloud_ceiling_per_1k_tok: dict[str, float]
    sku_yaml: str


def cost_per_1k_tokens_energy(joules_per_token: float,
                                usd_per_kwh: float = DEFAULT_USD_PER_KWH) -> float:
    """Energy-grounded floor: USD per 1000 generated tokens."""
    return joules_per_token * 1000 * _J_TO_KWH * usd_per_kwh


def cost_per_1k_tokens_cloud(seconds_per_token: float,
                               usd_per_hour_sku: float) -> float:
    """Cloud-rental ceiling for one SKU."""
    return (usd_per_hour_sku / 3600.0) * seconds_per_token * 1000


def quote(joules_per_token: float, seconds_per_token: float,
          usd_per_kwh: float = DEFAULT_USD_PER_KWH,
          sku_yaml: Path | None = None) -> CostQuote:
    sku_path = Path(sku_yaml) if sku_yaml else SKUS_YAML
    skus = yaml.safe_load(sku_path.read_text()) if sku_path.exists() else {}
    cloud = {
        name: cost_per_1k_tokens_cloud(seconds_per_token, info["usd_per_hour"])
        for name, info in skus.get("skus", {}).items()
    }
    return CostQuote(
        energy_floor_per_1k_tok=cost_per_1k_tokens_energy(joules_per_token, usd_per_kwh),
        cloud_ceiling_per_1k_tok=cloud,
        sku_yaml=str(sku_path),
    )


def main() -> int:
    """Demo: quote a hypothetical workload."""
    j_per_tok = 0.025  # 25 mJ/token, plausible for a small LLM on M5 Max
    s_per_tok = 0.008  # 8 ms/token, matches iter-6 fp16 baseline
    q = quote(j_per_tok, s_per_tok)
    print(f"j_per_tok = {j_per_tok}, s_per_tok = {s_per_tok}, $/kWh = {DEFAULT_USD_PER_KWH}")
    print(f"  energy_floor:  ${q.energy_floor_per_1k_tok:.6f} / 1k tokens")
    for sku, price in q.cloud_ceiling_per_1k_tok.items():
        print(f"  cloud ({sku}): ${price:.4f} / 1k tokens")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
