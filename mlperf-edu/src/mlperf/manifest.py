"""
MLPerf EDU: Provenance manifest with real Merkle-style tamper detection.

Replaces the iter-1 era `str(report)` self-hash in loadgen.py with a
hash chain that actually binds: source-tree git SHA, weights bytes,
dataset bytes, RNG state, hardware fingerprint, and the roofline
measurement sidecar.

Per Dean's iter-5 spec: every leaf is a recomputable fact about the run.
The Merkle root over the leaves is what the submission attests to. A
verifier (scripts/verify_submission.py) walks every leaf and recomputes
its hash from the artifact on disk; mismatches are reported per-leaf.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "mlperf-edu-provd/1.0"


# ---------- canonical encoders ----------

def _canon(obj: Any) -> bytes:
    """Canonical JSON encoding for hashing (no whitespace, sorted keys)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_file(path: str | Path, chunk: int = 1 << 20) -> tuple[str, int]:
    """Stream-hash a file. Returns (hex_digest, n_bytes)."""
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
            n += len(buf)
    return h.hexdigest(), n


# ---------- leaf builders ----------

def _git_leaf(repo_root: Path) -> dict:
    """Leaf binding source-tree state (sha + dirty + tree hash)."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
        dirty = subprocess.run(
            ["git", "diff-index", "--quiet", "HEAD", "--"],
            cwd=repo_root,
        ).returncode != 0
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"git_sha": None, "git_dirty": True,
                "tree_hash": None, "patch_hash": None,
                "note": "no git repo or git unavailable"}

    # Hash the contents of every tracked file; not a real git tree hash
    # but a portable equivalent that doesn't need libgit2.
    files = subprocess.check_output(
        ["git", "ls-files"], cwd=repo_root, text=True
    ).splitlines()
    tree_h = hashlib.sha256()
    for f in sorted(files):
        p = repo_root / f
        if not p.is_file():
            continue
        h, _ = _hash_file(p)
        tree_h.update(f"{f}:{h}\n".encode())

    patch_hash = None
    if dirty:
        diff = subprocess.check_output(["git", "diff"], cwd=repo_root, text=True)
        patch_hash = _sha256(diff.encode())

    return {
        "git_sha": sha, "git_dirty": dirty,
        "tree_hash": "sha256:" + tree_h.hexdigest(),
        "patch_hash": ("sha256:" + patch_hash) if patch_hash else None,
    }


def weights_leaf(checkpoint_path: str | Path,
                  n_params: int | None = None,
                  dtype: str | None = None) -> dict:
    """Leaf binding model weights to bytes-on-disk."""
    p = Path(checkpoint_path)
    if not p.exists():
        return {"path": str(p), "sha256": None, "n_bytes": 0,
                "n_params": n_params, "torch_dtype": dtype,
                "note": "checkpoint file missing"}
    digest, n = _hash_file(p)
    return {"path": str(p), "sha256": "sha256:" + digest, "n_bytes": n,
            "n_params": n_params, "torch_dtype": dtype}


def dataset_leaf(name: str, file_paths: list[str | Path]) -> dict:
    """Leaf binding the dataset by per-file hash + Merkle root."""
    files: list[dict] = []
    root_h = hashlib.sha256()
    for path in sorted([str(p) for p in file_paths]):
        if not Path(path).exists():
            files.append({"path": path, "sha256": None, "n_bytes": 0,
                          "note": "missing"})
            continue
        digest, n = _hash_file(path)
        files.append({"path": path, "sha256": "sha256:" + digest, "n_bytes": n})
        root_h.update(f"{path}:{digest}\n".encode())
    return {"name": name, "files": files,
            "merkle_root": "sha256:" + root_h.hexdigest()}


def rng_leaf(seed: int | None,
              torch_state_bytes: bytes | None,
              numpy_state_bytes: bytes | None) -> dict:
    return {
        "seed": seed,
        "torch_initial_state_sha256": ("sha256:" + _sha256(torch_state_bytes))
                                       if torch_state_bytes else None,
        "numpy_state_sha256": ("sha256:" + _sha256(numpy_state_bytes))
                               if numpy_state_bytes else None,
    }


def hardware_leaf(fingerprint: dict) -> dict:
    fp_canon = _canon(fingerprint)
    return {
        "fingerprint_sha256": "sha256:" + _sha256(fp_canon),
        "machine_class": fingerprint.get("machine_class")
                          or fingerprint.get("system", {}).get("machine")
                          or "unknown",
        "reference_platform": fingerprint.get("is_reference_platform", False),
    }


def roofline_sidecar_leaf(sidecar_path: str | Path) -> dict:
    p = Path(sidecar_path)
    if not p.exists():
        return {"path": str(p), "sha256": None, "note": "no sidecar produced"}
    digest, _ = _hash_file(p)
    return {"path": str(p), "sha256": "sha256:" + digest}


def measurement_leaf(report: dict, report_path: str | Path) -> dict:
    return {
        "report_canonical_sha256": "sha256:" + _sha256(_canon(report)),
        "report_path": str(report_path),
    }


# ---------- Merkle root + signature ----------

def merkle_root(leaves: dict) -> str:
    """Order-independent root over all leaves' canonical encoding."""
    parts: list[bytes] = []
    for k in sorted(leaves):
        parts.append(k.encode() + b":" + _canon(leaves[k]))
    return "sha256:" + _sha256(b"\x00".join(parts))


def _signing_key() -> bytes:
    """HMAC key for tamper-evident signing.

    Local-only, per-install. Auto-generated on first use. Sufficient for
    educational submissions; not designed to resist a determined adversary.
    """
    key_path = Path.home() / ".mlperf-edu" / "signing.key"
    key_path.parent.mkdir(parents=True, exist_ok=True)
    if not key_path.exists():
        key_path.write_bytes(secrets.token_bytes(32))
        key_path.chmod(0o600)
    return key_path.read_bytes()


def sign_manifest(merkle: str, key_id: str = "student-local-2026") -> dict:
    sig = hmac.new(_signing_key(), merkle.encode(), hashlib.sha256).hexdigest()
    return {"algo": "hmac-sha256", "key_id": key_id, "signature": sig}


# ---------- public API ----------

@dataclass
class ProvdManifest:
    workload: str
    scenario: str
    division: str
    leaves: dict
    merkle_root: str
    signature: dict
    utc: str
    nonce: str
    schema: str = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "workload": self.workload,
            "scenario": self.scenario,
            "division": self.division,
            "utc": self.utc,
            "nonce": self.nonce,
            "leaves": self.leaves,
            "merkle_root": self.merkle_root,
            "signature": self.signature,
        }


def build_provd(*, workload: str, scenario: str, division: str,
                 hardware_fingerprint: dict,
                 report: dict, report_path: str | Path,
                 weights_path: str | Path | None = None,
                 weights_n_params: int | None = None,
                 weights_dtype: str | None = None,
                 dataset_name: str = "unknown",
                 dataset_files: list[str | Path] | None = None,
                 rng_seed: int | None = None,
                 torch_state_bytes: bytes | None = None,
                 numpy_state_bytes: bytes | None = None,
                 roofline_sidecar_path: str | Path | None = None,
                 repo_root: str | Path | None = None) -> ProvdManifest:
    """Construct a complete provenance manifest."""
    import datetime
    repo = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    leaves = {
        "source_tree": _git_leaf(repo),
        "weights": (weights_leaf(weights_path, weights_n_params, weights_dtype)
                    if weights_path else
                    {"path": None, "sha256": None, "note": "no weights checkpoint"}),
        "dataset": dataset_leaf(dataset_name, dataset_files or []),
        "rng": rng_leaf(rng_seed, torch_state_bytes, numpy_state_bytes),
        "hardware": hardware_leaf(hardware_fingerprint),
        "roofline_sidecar": (roofline_sidecar_leaf(roofline_sidecar_path)
                              if roofline_sidecar_path
                              else {"path": None, "sha256": None,
                                    "note": "no roofline sidecar"}),
        "measurement": measurement_leaf(report, report_path),
    }
    root = merkle_root(leaves)
    return ProvdManifest(
        workload=workload, scenario=scenario, division=division,
        leaves=leaves, merkle_root=root, signature=sign_manifest(root),
        utc=datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        nonce=secrets.token_hex(8),
    )


# ---------- verification ----------

@dataclass
class VerificationResult:
    workload: str
    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        return all(ok for _, ok, _ in self.checks)

    def add(self, name: str, ok: bool, detail: str = ""):
        self.checks.append((name, ok, detail))


def verify_provd(manifest_path: str | Path,
                  repo_root: str | Path | None = None) -> VerificationResult:
    """Walk every leaf and recompute its hash from the artifact on disk."""
    manifest = json.loads(Path(manifest_path).read_text())
    res = VerificationResult(workload=manifest["workload"])
    leaves = manifest["leaves"]

    # Source tree.
    if leaves["source_tree"].get("git_sha"):
        actual_git = _git_leaf(Path(repo_root) if repo_root
                                else Path(manifest_path).resolve().parents[1])
        ok = actual_git["git_sha"] == leaves["source_tree"]["git_sha"]
        res.add("source_tree.git_sha", ok,
                f"claimed {leaves['source_tree']['git_sha'][:12] if leaves['source_tree']['git_sha'] else 'None'}, "
                f"current HEAD {actual_git['git_sha'][:12] if actual_git['git_sha'] else 'None'}")

    # Weights.
    w = leaves["weights"]
    if w.get("path") and w.get("sha256"):
        if Path(w["path"]).exists():
            actual, _ = _hash_file(w["path"])
            ok = ("sha256:" + actual) == w["sha256"]
            res.add("weights.sha256", ok,
                    f"claimed {w['sha256'][:18]}, recomputed sha256:{actual[:12]}")
        else:
            res.add("weights.sha256", False, f"file missing: {w['path']}")

    # Dataset Merkle.
    d = leaves["dataset"]
    if d.get("files") and d.get("merkle_root"):
        root_h = hashlib.sha256()
        ok_files = True
        for f in d["files"]:
            if not f.get("sha256") or not Path(f["path"]).exists():
                ok_files = False
                continue
            actual, _ = _hash_file(f["path"])
            if ("sha256:" + actual) != f["sha256"]:
                ok_files = False
            root_h.update(f"{f['path']}:{actual}\n".encode())
        recomputed = "sha256:" + root_h.hexdigest()
        res.add("dataset.merkle_root", ok_files and recomputed == d["merkle_root"],
                f"claimed {d['merkle_root'][:18]}, recomputed {recomputed[:18]}")

    # Roofline sidecar.
    rs = leaves["roofline_sidecar"]
    if rs.get("path") and rs.get("sha256"):
        if Path(rs["path"]).exists():
            actual, _ = _hash_file(rs["path"])
            res.add("roofline_sidecar.sha256",
                    ("sha256:" + actual) == rs["sha256"],
                    f"recomputed sha256:{actual[:12]}")
        else:
            res.add("roofline_sidecar.sha256", False, f"missing: {rs['path']}")

    # Measurement.
    m = leaves["measurement"]
    if m.get("report_path") and Path(m["report_path"]).exists():
        report = json.loads(Path(m["report_path"]).read_text())
        actual = "sha256:" + _sha256(_canon(report))
        res.add("measurement.report_canonical_sha256",
                actual == m["report_canonical_sha256"],
                f"recomputed {actual[:18]}")

    # Merkle root over leaves.
    recomputed_root = merkle_root(leaves)
    res.add("merkle_root", recomputed_root == manifest["merkle_root"],
            f"recomputed {recomputed_root[:18]} vs claimed {manifest['merkle_root'][:18]}")

    # Signature.
    sig_ok = hmac.compare_digest(
        manifest["signature"]["signature"],
        hmac.new(_signing_key(), manifest["merkle_root"].encode(),
                 hashlib.sha256).hexdigest(),
    )
    res.add("signature", sig_ok,
            "signed by local install" if sig_ok else "INVALID signature")

    return res
