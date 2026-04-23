#!/usr/bin/env python3
"""
Iter-9 smoke test for the data-quality lab.

Verifies the label-flip injection produces a measurable per-movie drift
without changing the global rating distribution by much. This is the
"silent accuracy drop" pedagogy made measurable.

Gate (dimensionless):
  per_movie_drift_max / global_mean_drift > 5.0
  i.e. the bug shows up >5x more strongly in the per-movie slice than
  in the global summary stat — exactly the dashboard-blind regime.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from labs.data_quality.inject import inject_label_flip, detect_label_flip


def synthetic_ratings(n: int = 10000, n_movies: int = 1682, seed: int = 42):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        u = rng.randint(0, 942)
        m = rng.randint(0, n_movies - 1)
        r = rng.randint(1, 5)
        t = rng.randint(0, 10**9)
        rows.append((u, m, r, t))
    return rows


def main() -> int:
    print("Iter-9 smoke: data quality (label-flip injection)")
    print("-" * 60)
    baseline = synthetic_ratings()
    print(f"  baseline rows: {len(baseline)}")
    base_global_mean = sum(r for _, _, r, _ in baseline) / len(baseline)
    print(f"  baseline global mean rating: {base_global_mean:.3f}")

    # Pick the first 17 movies as "Drama" (1% of 1682).
    drama_ids = list(range(17))
    corrupted, stats = inject_label_flip(baseline, drama_ids)
    print(f"  flipped {stats['n_flipped']} ratings ({stats['fraction_flipped']*100:.2f}% of dataset)")
    corrupt_global_mean = sum(r for _, _, r, _ in corrupted) / len(corrupted)
    global_drift = abs(corrupt_global_mean - base_global_mean)
    print(f"  global mean rating after: {corrupt_global_mean:.3f}  (drift: {global_drift:.4f})")

    detect = detect_label_flip(baseline, corrupted)
    print(f"  max per-movie drift: {detect['max_per_movie_drift']:.3f}")
    print(f"  mean per-movie drift: {detect['mean_per_movie_drift']:.3f}")
    print(f"  movies with >0.5 drift: {detect['n_movies_with_significant_drift']}")

    detection_amplification = (detect["max_per_movie_drift"] /
                                 max(global_drift, 1e-6))
    print()
    print(f"detection amplification (per-movie / global): {detection_amplification:.1f}x  (gate > 5x)")
    if detection_amplification > 5.0:
        print("ITER-9 SMOKE: PASS")
        print(f"  Headline: \"A 1% label-flip moves global mean by {global_drift:.4f} (invisible) "
              f"but per-movie drift by {detect['max_per_movie_drift']:.2f} ({detection_amplification:.0f}x). "
              f"Slice-level monitoring catches what dashboards miss.\"")
        return 0
    print("ITER-9 SMOKE: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
