"""
MLPerf EDU: Provenance manifest with real Merkle-style tamper detection.

Replaces the iter-1 era `str(report)` self-hash in loadgen.py with a
hash chain that actually binds: source-tree git SHA, weights bytes,
dataset bytes, RNG state, hardware fingerprint, and the roofline
measurement sidecar.

Every leaf is a recomputable fact about the run. The Merkle root and its
domain-separated integrity digest provide tamper detection, not producer
authentication. A verifier walks every leaf and recomputes its hash from the
artifact on disk; mismatches are reported per leaf.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "mlperf-edu-provd/1.1"
INTEGRITY_DIGEST_ALGO = "sha256-domain-separated-digest-v1"
INTEGRITY_DIGEST_DOMAIN = "mlperf-edu-provd-integrity-v1"

# Backward-compatible identifier for manifests emitted before schema 1.1. Those
# manifests called a public checksum a "signature" even though no secret or
# public/private key was involved. Verification accepts them, but new manifests
# accurately describe the value as an unauthenticated integrity digest.
LEGACY_PORTABLE_SIGNATURE_ALGO = "sha256-merkle-root-v1"
LEGACY_PORTABLE_SIGNATURE_DOMAIN = "mlperf-edu-provd-signature-v1"


# ---------- canonical encoders ----------


def _canon(obj: Any) -> bytes:
    """Canonical JSON encoding for hashing (no whitespace, sorted keys)."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


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
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = (
            subprocess.run(
                ["git", "diff-index", "--quiet", "HEAD", "--"],
                cwd=repo_root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            != 0
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {
            "git_sha": None,
            "git_dirty": True,
            "tree_hash": None,
            "patch_hash": None,
            "note": "no git repo or git unavailable",
        }

    # Hash the contents of every tracked file; not a real git tree hash
    # but a portable equivalent that doesn't need libgit2.
    files = subprocess.check_output(
        ["git", "ls-files"],
        cwd=repo_root,
        text=True,
        stderr=subprocess.DEVNULL,
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
        diff = subprocess.check_output(
            ["git", "diff"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        patch_hash = _sha256(diff.encode())

    return {
        "git_sha": sha,
        "git_dirty": dirty,
        "tree_hash": "sha256:" + tree_h.hexdigest(),
        "patch_hash": ("sha256:" + patch_hash) if patch_hash else None,
    }


def weights_leaf(
    checkpoint_path: str | Path, n_params: int | None = None, dtype: str | None = None
) -> dict:
    """Leaf binding model weights to bytes-on-disk."""
    p = Path(checkpoint_path)
    if not p.exists():
        return {
            "path": str(p),
            "sha256": None,
            "n_bytes": 0,
            "n_params": n_params,
            "torch_dtype": dtype,
            "note": "checkpoint file missing",
        }
    digest, n = _hash_file(p)
    return {
        "path": str(p),
        "sha256": "sha256:" + digest,
        "n_bytes": n,
        "n_params": n_params,
        "torch_dtype": dtype,
    }


def dataset_leaf(name: str, file_paths: list[str | Path]) -> dict:
    """Leaf binding the dataset by per-file hash + Merkle root."""
    files: list[dict] = []
    root_h = hashlib.sha256()
    for path in sorted([str(p) for p in file_paths]):
        if not Path(path).exists():
            files.append(
                {"path": path, "sha256": None, "n_bytes": 0, "note": "missing"}
            )
            continue
        digest, n = _hash_file(path)
        files.append({"path": path, "sha256": "sha256:" + digest, "n_bytes": n})
        root_h.update(f"{path}:{digest}\n".encode())
    return {"name": name, "files": files, "merkle_root": "sha256:" + root_h.hexdigest()}


def rng_leaf(
    seed: int | None, torch_state_bytes: bytes | None, numpy_state_bytes: bytes | None
) -> dict:
    return {
        "seed": seed,
        "torch_initial_state_sha256": ("sha256:" + _sha256(torch_state_bytes))
        if torch_state_bytes
        else None,
        "numpy_state_sha256": ("sha256:" + _sha256(numpy_state_bytes))
        if numpy_state_bytes
        else None,
    }


def hardware_leaf(fingerprint: dict) -> dict:
    fp_canon = _canon(fingerprint)
    return {
        "fingerprint": fingerprint,
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
    digest, n = _hash_file(p)
    return {"path": str(p), "sha256": "sha256:" + digest, "n_bytes": n}


def measurement_leaf(
    report: dict,
    report_path: str | Path,
    *,
    report_bytes: bytes | None = None,
) -> dict:
    """Bind both report semantics and exact serialized bytes.

    ``report_canonical_sha256`` catches semantic changes independent of JSON
    formatting. ``report_file_sha256`` also covers the exact artifact carried in
    a submission package. ``report_bytes`` supports archive-specific reports
    that are assembled in memory before they are written.
    """
    if report_bytes is None:
        path = Path(report_path)
        report_bytes = path.read_bytes() if path.exists() and path.is_file() else None
    leaf = {
        "report_canonical_sha256": "sha256:" + _sha256(_canon(report)),
        "report_path": str(report_path),
    }
    if report_bytes is not None:
        leaf["report_file_sha256"] = "sha256:" + _sha256(report_bytes)
        leaf["n_bytes"] = len(report_bytes)
    return leaf


# ---------- Merkle root + integrity digest ----------


def merkle_root(leaves: dict) -> str:
    """Order-independent root over all leaves' canonical encoding."""
    parts: list[bytes] = []
    for k in sorted(leaves):
        parts.append(k.encode() + b":" + _canon(leaves[k]))
    return "sha256:" + _sha256(b"\x00".join(parts))


def _legacy_signing_key() -> bytes:
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


def _integrity_digest(merkle: str) -> str:
    return _sha256(f"{INTEGRITY_DIGEST_DOMAIN}:{merkle}".encode())


def integrity_record(merkle: str) -> dict:
    """Return a portable tamper-detection record.

    This is intentionally not called a signature. Anyone can recompute the
    digest, so it detects accidental or post-publication changes but does not
    authenticate who produced the manifest.
    """
    return {
        "type": "unauthenticated_digest",
        "algorithm": INTEGRITY_DIGEST_ALGO,
        "digest": _integrity_digest(merkle),
        "authenticated": False,
    }


# ---------- public API ----------


@dataclass
class ProvdManifest:
    workload: str
    scenario: str
    division: str
    leaves: dict
    merkle_root: str
    integrity: dict
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
            "integrity": self.integrity,
        }


def build_provd(
    *,
    workload: str,
    scenario: str,
    division: str,
    hardware_fingerprint: dict,
    report: dict,
    report_path: str | Path,
    weights_path: str | Path | None = None,
    weights_n_params: int | None = None,
    weights_dtype: str | None = None,
    dataset_name: str = "unknown",
    dataset_files: list[str | Path] | None = None,
    rng_seed: int | None = None,
    torch_state_bytes: bytes | None = None,
    numpy_state_bytes: bytes | None = None,
    roofline_sidecar_path: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> ProvdManifest:
    """Construct a complete provenance manifest."""
    import datetime

    repo = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    leaves = {
        "source_tree": _git_leaf(repo),
        "weights": (
            weights_leaf(weights_path, weights_n_params, weights_dtype)
            if weights_path
            else {"path": None, "sha256": None, "note": "no weights checkpoint"}
        ),
        "dataset": dataset_leaf(dataset_name, dataset_files or []),
        "rng": rng_leaf(rng_seed, torch_state_bytes, numpy_state_bytes),
        "hardware": hardware_leaf(hardware_fingerprint),
        "roofline_sidecar": (
            roofline_sidecar_leaf(roofline_sidecar_path)
            if roofline_sidecar_path
            else {"path": None, "sha256": None, "note": "no roofline sidecar"}
        ),
        "measurement": measurement_leaf(report, report_path),
    }
    root = merkle_root(leaves)
    return ProvdManifest(
        workload=workload,
        scenario=scenario,
        division=division,
        leaves=leaves,
        merkle_root=root,
        integrity=integrity_record(root),
        utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
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


def resolve_artifact_path(manifest_path: str | Path, artifact_path: str | Path) -> Path:
    """Resolve a manifest artifact path, including archive-relative paths."""
    path = Path(artifact_path)
    if path.is_absolute():
        return path
    return Path(manifest_path).resolve().parent / path


def verify_provd(
    manifest_path: str | Path, repo_root: str | Path | None = None
) -> VerificationResult:
    """Walk every leaf and recompute its hash from the artifact on disk."""
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    res = VerificationResult(workload=manifest["workload"])
    leaves = manifest["leaves"]

    # Source tree.
    if leaves["source_tree"].get("git_sha"):
        actual_git = _git_leaf(
            Path(repo_root) if repo_root else manifest_path.parents[1]
        )
        ok = actual_git["git_sha"] == leaves["source_tree"]["git_sha"]
        res.add(
            "source_tree.git_sha",
            ok,
            f"claimed {leaves['source_tree']['git_sha'][:12] if leaves['source_tree']['git_sha'] else 'None'}, "
            f"current HEAD {actual_git['git_sha'][:12] if actual_git['git_sha'] else 'None'}",
        )
        if leaves["source_tree"].get("tree_hash"):
            res.add(
                "source_tree.tree_hash",
                actual_git.get("tree_hash") == leaves["source_tree"].get("tree_hash"),
                f"claimed {str(leaves['source_tree'].get('tree_hash'))[:18]}, "
                f"recomputed {str(actual_git.get('tree_hash'))[:18]}",
            )
        if leaves["source_tree"].get("git_dirty"):
            res.add(
                "source_tree.patch_hash",
                actual_git.get("patch_hash") == leaves["source_tree"].get("patch_hash"),
                f"claimed {str(leaves['source_tree'].get('patch_hash'))[:18]}, "
                f"recomputed {str(actual_git.get('patch_hash'))[:18]}",
            )

        # Closed-division rejection of dirty trees (Dean's iter-5 sign-off).
        # Open division allows dirty trees with patch_hash as a courtesy.
        if manifest.get("division") == "closed" and leaves["source_tree"].get(
            "git_dirty"
        ):
            res.add(
                "source_tree.closed_division_clean",
                False,
                "division=closed but submission was generated from a dirty tree "
                "(uncommitted changes). Closed-division submissions must come "
                "from a clean working tree.",
            )

    # Weights.
    w = leaves["weights"]
    if w.get("path") and not w.get("sha256"):
        res.add(
            "weights.sha256", False, f"digest absent for declared file: {w['path']}"
        )
    if w.get("sha256") and not w.get("path"):
        res.add("weights.path", False, "digest declared without an artifact path")
    if w.get("path") and w.get("sha256"):
        weights_path = resolve_artifact_path(manifest_path, w["path"])
        if weights_path.exists():
            actual, n_bytes = _hash_file(weights_path)
            ok = ("sha256:" + actual) == w["sha256"]
            res.add(
                "weights.sha256",
                ok,
                f"claimed {w['sha256'][:18]}, recomputed sha256:{actual[:12]}",
            )
            if w.get("n_bytes") is not None:
                res.add(
                    "weights.n_bytes",
                    n_bytes == w["n_bytes"],
                    f"claimed {w['n_bytes']}, recomputed {n_bytes}",
                )
        else:
            res.add("weights.sha256", False, f"file missing: {w['path']}")

    # Dataset Merkle.
    d = leaves["dataset"]
    if d.get("files") and not d.get("merkle_root"):
        res.add(
            "dataset.merkle_root", False, "dataset files declared without a Merkle root"
        )
    if d.get("files") and d.get("merkle_root"):
        root_h = hashlib.sha256()
        ok_files = True
        for index, f in enumerate(d["files"]):
            dataset_path = resolve_artifact_path(manifest_path, f["path"])
            if not f.get("sha256") or not dataset_path.exists():
                ok_files = False
                res.add(
                    f"dataset.files[{index}].sha256",
                    False,
                    f"file missing or digest absent: {f.get('path')}",
                )
                continue
            actual, n_bytes = _hash_file(dataset_path)
            digest_ok = ("sha256:" + actual) == f["sha256"]
            res.add(
                f"dataset.files[{index}].sha256",
                digest_ok,
                f"claimed {f['sha256'][:18]}, recomputed sha256:{actual[:12]}",
            )
            if not digest_ok:
                ok_files = False
            if f.get("n_bytes") is not None:
                size_ok = n_bytes == f["n_bytes"]
                res.add(
                    f"dataset.files[{index}].n_bytes",
                    size_ok,
                    f"claimed {f['n_bytes']}, recomputed {n_bytes}",
                )
                ok_files = ok_files and size_ok
            root_h.update(f"{f['path']}:{actual}\n".encode())
        recomputed = "sha256:" + root_h.hexdigest()
        res.add(
            "dataset.merkle_root",
            ok_files and recomputed == d["merkle_root"],
            f"claimed {d['merkle_root'][:18]}, recomputed {recomputed[:18]}",
        )

    # Roofline sidecar.
    rs = leaves["roofline_sidecar"]
    if rs.get("path") and not rs.get("sha256"):
        res.add(
            "roofline_sidecar.sha256",
            False,
            f"digest absent for declared file: {rs['path']}",
        )
    if rs.get("path") and rs.get("sha256"):
        roofline_path = resolve_artifact_path(manifest_path, rs["path"])
        if roofline_path.exists():
            actual, n_bytes = _hash_file(roofline_path)
            res.add(
                "roofline_sidecar.sha256",
                ("sha256:" + actual) == rs["sha256"],
                f"recomputed sha256:{actual[:12]}",
            )
            if rs.get("n_bytes") is not None:
                res.add(
                    "roofline_sidecar.n_bytes",
                    n_bytes == rs["n_bytes"],
                    f"claimed {rs['n_bytes']}, recomputed {n_bytes}",
                )
        else:
            res.add("roofline_sidecar.sha256", False, f"missing: {rs['path']}")

    # Hardware metadata. This validates internal consistency and keeps the full
    # machine description inspectable; it does not independently attest that the
    # run occurred on the described hardware.
    hardware = leaves["hardware"]
    fingerprint = hardware.get("fingerprint")
    if fingerprint is not None:
        actual = "sha256:" + _sha256(_canon(fingerprint))
        res.add(
            "hardware.fingerprint_sha256",
            actual == hardware.get("fingerprint_sha256"),
            f"recomputed {actual[:18]}",
        )
    elif manifest.get("schema") == SCHEMA_VERSION:
        res.add(
            "hardware.fingerprint",
            False,
            "schema 1.1 requires inspectable hardware metadata",
        )

    # Measurement.
    m = leaves["measurement"]
    if not m.get("report_path"):
        res.add("measurement.report_path", False, "measurement report path is absent")
    if not m.get("report_canonical_sha256"):
        res.add(
            "measurement.report_canonical_sha256", False, "measurement digest is absent"
        )
    if manifest.get("schema") == SCHEMA_VERSION and not m.get("report_file_sha256"):
        res.add(
            "measurement.report_file_sha256",
            False,
            "schema 1.1 requires an exact report-file digest",
        )
    report_path = (
        resolve_artifact_path(manifest_path, m["report_path"])
        if m.get("report_path")
        else None
    )
    if report_path and report_path.exists():
        report_bytes = report_path.read_bytes()
        report = json.loads(report_bytes)
        actual = "sha256:" + _sha256(_canon(report))
        res.add(
            "measurement.report_canonical_sha256",
            actual == m["report_canonical_sha256"],
            f"recomputed {actual[:18]}",
        )
        if m.get("report_file_sha256"):
            file_digest = "sha256:" + _sha256(report_bytes)
            res.add(
                "measurement.report_file_sha256",
                file_digest == m["report_file_sha256"],
                f"recomputed {file_digest[:18]}",
            )
        if m.get("n_bytes") is not None:
            res.add(
                "measurement.n_bytes",
                len(report_bytes) == m["n_bytes"],
                f"claimed {m['n_bytes']}, recomputed {len(report_bytes)}",
            )
    elif m.get("report_path"):
        res.add(
            "measurement.report_canonical_sha256", False, f"missing: {m['report_path']}"
        )

    # Merkle root over leaves.
    recomputed_root = merkle_root(leaves)
    res.add(
        "merkle_root",
        recomputed_root == manifest["merkle_root"],
        f"recomputed {recomputed_root[:18]} vs claimed {manifest['merkle_root'][:18]}",
    )

    # Integrity record. Schema 1.1 accurately labels this as an unauthenticated
    # digest. Older manifests using the misleading signature label remain
    # verifiable for compatibility.
    integrity = manifest.get("integrity") or {}
    if integrity:
        algo = integrity.get("algorithm")
        if algo == INTEGRITY_DIGEST_ALGO:
            expected = _integrity_digest(manifest["merkle_root"])
            sig_ok = hmac.compare_digest(str(integrity.get("digest", "")), expected)
            detail = (
                "valid unauthenticated integrity digest (tamper detection only; origin not authenticated)"
                if sig_ok
                else "INVALID unauthenticated integrity digest"
            )
        else:
            sig_ok = False
            detail = f"unsupported integrity digest algorithm: {algo}"
        res.add("integrity.digest", sig_ok, detail)
        return res

    signature = manifest.get("signature") or {}
    algo = signature.get("algo")
    if algo == LEGACY_PORTABLE_SIGNATURE_ALGO:
        expected = _sha256(
            f"{LEGACY_PORTABLE_SIGNATURE_DOMAIN}:{manifest['merkle_root']}".encode()
        )
        sig_ok = hmac.compare_digest(str(signature.get("signature", "")), expected)
        detail = (
            "valid legacy unauthenticated digest" if sig_ok else "INVALID legacy digest"
        )
    elif algo == "hmac-sha256":
        sig_ok = hmac.compare_digest(
            str(signature.get("signature", "")),
            hmac.new(
                _legacy_signing_key(), manifest["merkle_root"].encode(), hashlib.sha256
            ).hexdigest(),
        )
        detail = (
            "legacy local install signature"
            if sig_ok
            else "INVALID legacy local signature"
        )
    else:
        sig_ok = False
        detail = f"unsupported signature algorithm: {algo}"
    res.add("legacy_signature", sig_ok, detail)

    return res
