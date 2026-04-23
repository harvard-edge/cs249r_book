"""
Iter-9 (Huyen): Data quality bug injection for the MovieLens DLRM workload.

Two pedagogically-clear bugs:
  - Pick A: label flip on a single genre (1% of dataset).
  - Pick B: future timestamps on 5% of rows.

Both produce silent accuracy drops without obvious schema/row-count
changes. The lab teaches "your dashboard is green, your model is broken."
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable


def inject_label_flip(ratings: list[tuple], flip_genre_movie_ids: Iterable[int],
                       seed: int = 42) -> tuple[list, dict]:
    """Flip ratings 4->1, 5->2 on rows whose movie_id is in `flip_genre_movie_ids`.

    Args:
        ratings: list of (user_id, movie_id, rating, timestamp) tuples.
        flip_genre_movie_ids: set of movie IDs in the target genre.

    Returns:
        (corrupted_ratings, stats) where stats reports n_flipped / total.
    """
    flip_set = set(flip_genre_movie_ids)
    rng = random.Random(seed)
    corrupted = []
    n_flipped = 0
    for u, m, r, t in ratings:
        if m in flip_set and r >= 4:
            new_r = 1 if r == 4 else 2
            corrupted.append((u, m, new_r, t))
            n_flipped += 1
        else:
            corrupted.append((u, m, r, t))
    return corrupted, {
        "bug": "label_flip_on_genre",
        "n_flipped": n_flipped,
        "n_total": len(ratings),
        "fraction_flipped": n_flipped / max(len(ratings), 1),
    }


def inject_future_timestamps(ratings: list[tuple], fraction: float = 0.05,
                                shift_years: int = 10, seed: int = 42) -> tuple[list, dict]:
    """Shift `fraction` of timestamps forward by `shift_years` years."""
    rng = random.Random(seed)
    shift_seconds = shift_years * 365 * 24 * 3600
    n_shift = int(fraction * len(ratings))
    indices_to_shift = set(rng.sample(range(len(ratings)), n_shift))
    corrupted = []
    for i, (u, m, r, t) in enumerate(ratings):
        if i in indices_to_shift:
            corrupted.append((u, m, r, t + shift_seconds))
        else:
            corrupted.append((u, m, r, t))
    return corrupted, {
        "bug": "future_timestamps",
        "n_shifted": n_shift,
        "n_total": len(ratings),
        "fraction_shifted": fraction,
        "shift_seconds": shift_seconds,
    }


def detect_label_flip(ratings_baseline: list[tuple],
                       ratings_corrupted: list[tuple]) -> dict:
    """Detect a label-flip bug by per-genre rating-distribution drift.

    Returns the genre-averaged absolute mean-rating drift.
    """
    from collections import defaultdict
    by_movie_baseline = defaultdict(list)
    by_movie_corrupt = defaultdict(list)
    for u, m, r, t in ratings_baseline:
        by_movie_baseline[m].append(r)
    for u, m, r, t in ratings_corrupted:
        by_movie_corrupt[m].append(r)
    drifts = []
    n_changed_movies = 0
    for m in by_movie_baseline:
        b = sum(by_movie_baseline[m]) / len(by_movie_baseline[m])
        c = sum(by_movie_corrupt[m]) / max(len(by_movie_corrupt[m]), 1)
        drifts.append(abs(c - b))
        if abs(c - b) > 0.5:
            n_changed_movies += 1
    return {
        "max_per_movie_drift": max(drifts) if drifts else 0,
        "mean_per_movie_drift": sum(drifts) / max(len(drifts), 1),
        "n_movies_with_significant_drift": n_changed_movies,
    }
