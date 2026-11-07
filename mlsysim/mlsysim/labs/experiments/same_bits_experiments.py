"""Same-Bits experiments: why temperature 0 does not give the same answer twice.

Every number here is computed, not modelled.  numpy's float16 and float32 are
IEEE 754 binary16 and binary32, rounded after every operation, so the bit
patterns below are the ones a processor that adds in the same order produces.
Each kernel fixes its reduction order explicitly instead of calling BLAS, whose
order is not specified.

The chain the lab reproduces, one step per part:

A. floating-point addition is not associative, so one sum has several answers;
B. a serving kernel picks its reduction split from the batch size, so the same
   row gives different bits alone and inside a batch;
C. near-tied logits turn those bits into different greedy (temperature-0) text;
D. a batch-invariant kernel fixes the split and removes the change, at the
   price of extra partial-sum work for large batches;
E. int8 x int8 products with exact int32 sums give the same bits in any order,
   at the price of agreement with the float model.

The language model is a seeded toy recurrent model (64 wide, 76 tokens): big
enough to have near-tied logits, small enough to decode in a browser.  The
split policy is a teaching fixture shaped like tuned GPU libraries (split the
reduction when the batch is too small to fill the processor), not a measured
policy of any named product.  Track hardware comes from the MLSysIM registry.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from mlsysim.hardware.registry import Hardware

MODEL_ID = "same_bits_experiments"

DTYPES = {"float16": np.float16, "float32": np.float32}
_UINT = {2: np.uint16, 4: np.uint32}
_SINT = {2: np.int16, 4: np.int32}

ORDERS = ("left to right", "pairwise tree", "8 blocks", "64 blocks", "reversed")
POLICIES = ("throughput", "invariant")
MODES = ("float", "integer")


# Track scenarios ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScenarioSpec:
    """Who needs the same bits on this track, and the batches its server sees."""

    track_id: str
    hardware: object
    stakeholder: str
    why_bits_matter: str
    batch_options: tuple[int, ...]
    default_batch: int
    dtype: str

    @property
    def device_name(self) -> str:
        return self.hardware.name

    @property
    def int8_rate(self):
        """The registry's int8 rate for this device, or None when it lists none."""
        return self.hardware.compute.precision_flops.get("int8")


TRACKS = ("tinyml", "mobile", "edge", "cloud")


def get_scenario(track_id: str) -> ScenarioSpec:
    if track_id == "cloud":
        return ScenarioSpec(
            "cloud", Hardware.Cloud.H100, "RL post-training lead",
            "The trainer recomputes the log-probabilities of every token the inference "
            "server sampled. If the server's bits depend on who else was in the batch, "
            "the two disagree and the policy gradient is computed for text the policy "
            "never produced.",
            (1, 2, 4, 8, 16, 32), 16, "float16",
        )
    if track_id == "edge":
        return ScenarioSpec(
            "edge", Hardware.Edge.RoboTaxi, "Safety and incident-review engineer",
            "After an incident, reviewers replay the logged sensor windows through the "
            "same model. If the replay batches windows differently from the vehicle, "
            "the replayed decision can differ from the one the vehicle made.",
            (1, 2, 4, 8), 4, "float16",
        )
    if track_id == "mobile":
        return ScenarioSpec(
            "mobile", Hardware.Mobile.iPhone15Pro, "Mobile product engineer",
            "The keyboard scores several candidate continuations together. Support "
            "reproduces a user's report by scoring one candidate alone, and sees a "
            "different suggestion from the one the user saw.",
            (1, 2, 4, 8), 4, "float16",
        )
    if track_id == "tinyml":
        return ScenarioSpec(
            "tinyml", Hardware.Tiny.OuraRing, "Wearable firmware engineer",
            "Firmware updates are validated by comparing the ring's outputs with a "
            "reference run on a laptop. A float build that groups sensor windows "
            "differently on the two machines fails that check without any bug.",
            (1, 2, 4, 8), 4, "float16",
        )
    raise KeyError(f"unknown track {track_id!r}; expected one of {TRACKS}")


# Bit-level helpers -------------------------------------------------------------------------

def digest(a: np.ndarray) -> str:
    """First 12 hex digits of the SHA-256 of an array's bytes: equal digests mean equal bits."""
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:12]


def ulp_distance(a, b) -> int:
    """Largest number of representable floats between matching elements (0 = same bits)."""
    a, b = np.asarray(a), np.asarray(b)
    itype = _SINT[a.dtype.itemsize]
    ia, ib = a.view(itype).astype(np.int64), b.view(itype).astype(np.int64)
    # Map sign-magnitude onto a monotone integer line so subtraction counts steps.
    top = np.int64(1) << (8 * a.dtype.itemsize - 1)
    ia = np.where(ia < 0, -top - ia, ia)
    ib = np.where(ib < 0, -top - ib, ib)
    return int(np.max(np.abs(ia - ib)))


# Part A: one sum, five orders --------------------------------------------------------------

def sum_sequential(x: np.ndarray):
    """Left to right, rounding after every addition (one thread's loop)."""
    return np.add.accumulate(x, dtype=x.dtype)[-1]


def sum_tree(x: np.ndarray):
    """Pairwise tree, the shape of a GPU block reduction."""
    while x.size > 1:
        if x.size % 2:
            x = np.concatenate([x, np.zeros(1, x.dtype)])
        x = (x[0::2] + x[1::2]).astype(x.dtype)
    return x[0]


def sum_blocked(x: np.ndarray, blocks: int):
    """`blocks` contiguous chunks summed in parallel, then the partials added in order."""
    parts = [sum_sequential(c) for c in np.array_split(x, blocks)]
    return sum_sequential(np.array(parts, dtype=x.dtype))


def sum_in_order(x: np.ndarray, order: str):
    if order == "left to right":
        return sum_sequential(x)
    if order == "pairwise tree":
        return sum_tree(x)
    if order == "8 blocks":
        return sum_blocked(x, 8)
    if order == "64 blocks":
        return sum_blocked(x, 64)
    if order == "reversed":
        return sum_sequential(x[::-1].copy())
    raise KeyError(f"unknown order {order!r}; expected one of {ORDERS}")


def addition_orders(n: int = 4096, dtype: str = "float32", seed: int = 0) -> dict:
    """The same n numbers summed in every order of ORDERS."""
    x = np.random.default_rng(seed).standard_normal(n).astype(DTYPES[dtype])
    exact = float(np.sum(x.astype(np.float64)))
    values = {order: sum_in_order(x, order) for order in ORDERS}
    ref = np.array([values["left to right"]])
    rows = [
        {"order": order, "value": float(v), "bits": digest(np.array([v])),
         "ulps_from_left_to_right": ulp_distance(np.array([v]), ref),
         "error_vs_exact": abs(float(v) - exact)}
        for order, v in values.items()
    ]
    return {"n": n, "dtype": dtype, "exact": exact, "rows": rows,
            "distinct": len({row["bits"] for row in rows})}


def order_snapshot(order: str, *, n: int = 4096, dtype: str = "float32", seed: int = 0) -> dict:
    data = addition_orders(n, dtype, seed)
    row = next(r for r in data["rows"] if r["order"] == order)
    return {"inputs": {"order": order, "n": n, "dtype": dtype, "seed": seed},
            "value": row["value"], "bits": row["bits"],
            "ulps_from_left_to_right": row["ulps_from_left_to_right"],
            "distinct_over_all_orders": data["distinct"]}


# Part B: the batch picks the kernel --------------------------------------------------------

def splits_for_batch(batch: int, policy: str) -> int:
    """'throughput': split the reduction when the batch is too small to fill the processor,
    as tuned GPU libraries do.  'invariant': one split whatever the batch, so a row's sums
    never change order."""
    if policy == "invariant":
        return 4
    if policy != "throughput":
        raise KeyError(f"unknown policy {policy!r}; expected one of {POLICIES}")
    return 4 if batch < 4 else (2 if batch < 16 else 1)


def matmul_ordered(a: np.ndarray, w: np.ndarray, splits: int, dtype) -> np.ndarray:
    """a @ w with the K dimension cut into `splits` chunks (split-K).  Each chunk accumulates
    left to right, then partial outputs are added in chunk order, rounding after every step."""
    a, w = a.astype(dtype), w.astype(dtype)
    partials = []
    for idx in np.array_split(np.arange(a.shape[1]), splits):
        acc = np.zeros((a.shape[0], w.shape[1]), dtype)
        for k in idx:
            acc = (acc + a[:, k:k + 1] * w[k:k + 1, :]).astype(dtype)
        partials.append(acc)
    out = partials[0]
    for p in partials[1:]:
        out = (out + p).astype(dtype)
    return out


def batch_contrast(batch: int, dtype: str = "float16", policy: str = "throughput",
                   seed: int = 1, k: int = 256, n: int = 128) -> dict:
    """Row 0 of a k x n projection, computed alone and inside a batch of `batch` rows."""
    rng = np.random.default_rng(seed)
    dt = DTYPES[dtype]
    w = (rng.standard_normal((k, n)) / np.sqrt(k)).astype(dt)
    rows = rng.standard_normal((batch, k)).astype(dt)
    alone = matmul_ordered(rows[:1], w, splits_for_batch(1, policy), dt)[0]
    together = matmul_ordered(rows, w, splits_for_batch(batch, policy), dt)[0]
    uint = _UINT[np.dtype(dt).itemsize]
    return {
        "inputs": {"batch": batch, "dtype": dtype, "policy": policy, "seed": seed},
        "splits_alone": splits_for_batch(1, policy),
        "splits_batched": splits_for_batch(batch, policy),
        "digest_alone": digest(alone), "digest_batched": digest(together),
        "outputs": n,
        "outputs_differing": int(np.sum(alone.view(uint) != together.view(uint))),
        "max_ulps": ulp_distance(alone, together),
        "max_abs": float(np.max(np.abs(alone.astype(np.float64) - together.astype(np.float64)))),
    }


# Parts C-E: a tiny language model, decoded greedily ----------------------------------------

SYLLABLES = [c + v for c in "bdfgklmnprstvz" for v in "aeiou"] + ["ka", "lo", "mi", "ru", "sa", "te"]


@dataclass
class TinyLM:
    """Seeded toy recurrent language model.  The weights are random; the arithmetic is real."""

    vocab: int = 76
    width: int = 64
    seed: int = 7

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.emb = rng.standard_normal((self.vocab, self.width)) * 0.8
        self.mix = rng.standard_normal((self.width, self.width)) / np.sqrt(self.width) * 1.1
        self.out = rng.standard_normal((self.width, self.vocab)) / np.sqrt(self.width) * 3.0


def _quantize(x: np.ndarray, axis: int):
    scale = np.max(np.abs(x), axis=axis, keepdims=True) / 127.0
    scale = np.where(scale == 0, 1.0, scale)
    return np.clip(np.rint(x / scale), -127, 127).astype(np.int8), scale


def logits(model: TinyLM, h: np.ndarray, mode: str, policy: str, dtype: str = "float16") -> np.ndarray:
    """Output projection for a batch of hidden states.
    'float': split-K float kernel whose split the batch policy chooses.
    'integer': int8 x int8 with exact int32 sums, so every split gives the same bits."""
    batch = h.shape[0]
    if mode == "integer":
        hq, hs = _quantize(h, axis=1)
        wq, ws = _quantize(model.out, axis=0)
        acc = np.zeros((batch, model.vocab), np.int32)
        for idx in np.array_split(np.arange(model.width), splits_for_batch(batch, policy)):
            acc += hq[:, idx].astype(np.int32) @ wq[idx, :].astype(np.int32)
        return acc.astype(np.float64) * hs * ws
    if mode != "float":
        raise KeyError(f"unknown mode {mode!r}; expected one of {MODES}")
    return matmul_ordered(h, model.out, splits_for_batch(batch, policy), DTYPES[dtype]).astype(np.float64)


def decode(model: TinyLM, prompts: list[int], steps: int, batch: int, mode: str, policy: str,
           dtype: str = "float16") -> list[list[int]]:
    """Greedy (temperature 0) decoding of every prompt, served `batch` at a time."""
    outs = []
    for start in range(0, len(prompts), batch):
        group = prompts[start:start + batch]
        tok = np.array(group)
        h = np.zeros((len(group), model.width))
        seqs = [[t] for t in group]
        for _ in range(steps):
            h = np.tanh(h * 0.5 + model.emb[tok] @ model.mix)
            tok = np.argmax(logits(model, h, mode, policy, dtype), axis=1)
            for s, t in zip(seqs, tok):
                s.append(int(t))
        outs.extend(seqs)
    return outs


def words(seq: list[int]) -> str:
    return " ".join("".join(SYLLABLES[t] for t in seq[i:i + 3]) for i in range(0, len(seq), 3))


def serving_contrast(batch: int, mode: str = "float", policy: str = "throughput",
                     dtype: str = "float16", users: int = 64, steps: int = 40, seed: int = 7) -> dict:
    """The same prompts at temperature 0, served one at a time and then `batch` at a time."""
    return copy.deepcopy(_serving_contrast(batch, mode, policy, dtype, users, steps, seed))


@lru_cache(maxsize=64)
def _serving_contrast(batch, mode, policy, dtype, users, steps, seed) -> dict:
    model = TinyLM(seed=seed)
    prompts = list(range(users))
    alone = decode(model, prompts, steps, 1, mode, policy, dtype)
    together = decode(model, prompts, steps, batch, mode, policy, dtype)
    changed = []
    for i, (a, b) in enumerate(zip(alone, together)):
        if a != b:
            first = next(j for j, (x, y) in enumerate(zip(a, b)) if x != y)
            changed.append({"user": i, "first_divergent_token": first,
                            "alone": words(a), "batched": words(b)})
    return {
        "inputs": {"batch": batch, "mode": mode, "policy": policy, "dtype": dtype,
                   "users": users, "steps": steps, "seed": seed},
        "users": users, "changed": len(changed), "examples": changed[:3],
        "splits_alone": splits_for_batch(1, policy),
        "splits_batched": splits_for_batch(batch, policy),
        "digest_alone": digest(np.array(alone)), "digest_batched": digest(np.array(together)),
    }


def integer_agreement(users: int = 64, steps: int = 40, seed: int = 7) -> dict:
    """What integer arithmetic costs: how often its greedy tokens match a float32 run."""
    return dict(_integer_agreement(users, steps, seed))


@lru_cache(maxsize=8)
def _integer_agreement(users, steps, seed) -> dict:
    model = TinyLM(seed=seed)
    prompts = list(range(users))
    ref = decode(model, prompts, steps, 1, "float", "invariant", "float32")
    q = decode(model, prompts, steps, 1, "integer", "invariant")
    same = sum(a == b for r, s in zip(ref, q) for a, b in zip(r[1:], s[1:]))
    return {"users": users, "steps": steps,
            "token_agreement": same / (users * steps),
            "first_token_agreement": sum(r[1] == s[1] for r, s in zip(ref, q)) / users,
            "identical_sequences": sum(r == s for r, s in zip(ref, q))}


def extra_reduction_adds(batch: int, policy: str, outputs: int = 128) -> int:
    """Additions spent combining split-K partials for one matmul: the price of a fixed split."""
    return (splits_for_batch(batch, policy) - 1) * batch * outputs


def contract_snapshot(batch: int, mode: str, policy: str, dtype: str = "float16") -> dict:
    """One serving contract at one batch: changed users, partial-sum work, and bit digests."""
    serve = serving_contrast(batch, mode=mode, policy=policy, dtype=dtype)
    return {**serve, "extra_reduction_adds": extra_reduction_adds(batch, policy)}
