"""The NCF split must be deterministic under a fixed seed.

The evaluation negatives are part of the metric, not an implementation detail:
HR@10 is scored against them, so a change in the draw sequence changes the
number the 0.635 gate is evaluated against. Memory work on the split has
already touched this function once and will again when the per-user rejection
loop is vectorized. These tests pin the invariant so that a change which alters
the sample cannot pass silently.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest

from mlperf.runners.ncf import _load_leave_one_out


@pytest.fixture
def ratings(tmp_path):
    """A small synthetic MovieLens-shaped file: userId, movieId, rating, timestamp."""
    path = tmp_path / "ratings.csv"
    # Each user interacts with a strict subset of the catalog, so unseen items
    # remain available to sample as evaluation negatives.
    rows = [
        (user, 1 + (user * 3 + offset) % 20, 4.0, 1000 + user * 50 + offset)
        for user in range(1, 13)
        for offset in range(8)
    ]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["userId", "movieId", "rating", "timestamp"])
        writer.writerows(rows)
    return path


def test_same_seed_reproduces_the_evaluation_negatives(ratings):
    first = _load_leave_one_out(ratings, negatives_per_user=5, seed=42)
    second = _load_leave_one_out(ratings, negatives_per_user=5, seed=42)
    assert np.array_equal(first["eval_negatives"], second["eval_negatives"])
    assert np.array_equal(first["test_items"], second["test_items"])
    assert np.array_equal(first["train_users"], second["train_users"])
    assert np.array_equal(first["train_items"], second["train_items"])


def test_a_different_seed_draws_different_negatives(ratings):
    """Guards against the sampler ignoring the seed and returning a constant."""
    a = _load_leave_one_out(ratings, negatives_per_user=5, seed=42)
    b = _load_leave_one_out(ratings, negatives_per_user=5, seed=43)
    assert not np.array_equal(a["eval_negatives"], b["eval_negatives"])


def test_held_out_item_never_appears_in_training(ratings):
    """The leakage check that made the first 999-negative result believable."""
    split = _load_leave_one_out(ratings, negatives_per_user=5, seed=42)
    train = set(zip(split["train_users"].tolist(), split["train_items"].tolist()))
    for user in range(split["n_users"]):
        assert (user, int(split["test_items"][user])) not in train


def test_evaluation_negatives_exclude_items_the_user_interacted_with(ratings):
    split = _load_leave_one_out(ratings, negatives_per_user=5, seed=42)
    seen = {user: set() for user in range(split["n_users"])}
    for user, item in zip(split["train_users"].tolist(), split["train_items"].tolist()):
        seen[user].add(item)
    for user in range(split["n_users"]):
        seen[user].add(int(split["test_items"][user]))
        drawn = set(split["eval_negatives"][user].tolist())
        assert not (drawn & seen[user]), f"user {user} was offered a seen item"


def test_split_does_not_retain_the_per_user_seen_sets(ratings):
    """Returning them held gigabytes alive for the whole training run."""
    split = _load_leave_one_out(ratings, negatives_per_user=5, seed=42)
    assert "seen" not in split


def test_a_user_who_has_seen_almost_everything_fails_loudly(tmp_path):
    """This configuration used to spin forever inside the rejection sampler."""
    path = tmp_path / "dense.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["userId", "movieId", "rating", "timestamp"])
        writer.writerows(
            (user, item, 4.0, 1000 + user * 10 + item)
            for user in range(1, 4)
            for item in range(1, 6)
        )
    with pytest.raises(ValueError, match="unseen items"):
        _load_leave_one_out(path, negatives_per_user=5, seed=42)
