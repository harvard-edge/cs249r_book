from __future__ import annotations

import hashlib
import gzip
import os
import shutil
import subprocess
import tarfile
import uuid
import warnings
import zipfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TINY_SHAKESPEARE_URL = "https://www.gutenberg.org/files/100/100-0.txt"
TINY_SHAKESPEARE_VERSION = "project-gutenberg-100-0-tiny-excerpt"
TINY_SHAKESPEARE_TARGET_CHARS = 1_115_394
KARPATHY_TINY_SHAKESPEARE_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
MOVIELENS_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
MNIST_SOURCE = "torchvision://MNIST"
FASHION_MNIST_SOURCE = "torchvision://FashionMNIST"
CIFAR100_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
CIFAR100_PICKLE_WARNING = r"dtype\(\): align should be passed as Python or NumPy boolean"


@dataclass(frozen=True)
class DatasetAsset:
    name: str
    root: Path
    files: tuple[Path, ...]
    sha256: str
    n_bytes: int
    source: str


@dataclass(frozen=True)
class AssetDossier:
    id: str
    asset_type: str
    display_name: str
    source_url: str
    citation: str
    license: str
    license_spdx: str | None
    license_status: str
    terms_summary: str
    public_result_use: str
    public_release_status: str
    public_release_policy: str
    release_next_step: str | None = None
    license_evidence_url: str | None = None
    attribution: str | None = None
    version: str | None = None
    expected_download_bytes: int | None = None
    expected_unpacked_bytes: int | None = None
    hash_policy: str = "Reports and provenance manifests record computed hashes from the local fetched files."

    def to_dict(self) -> dict[str, Any]:
        data = {
            "id": self.id,
            "type": self.asset_type,
            "display_name": self.display_name,
            "source_url": self.source_url,
            "citation": self.citation,
            "license": self.license,
            "license_spdx": self.license_spdx,
            "license_status": self.license_status,
            "terms_summary": self.terms_summary,
            "public_result_use": self.public_result_use,
            "public_release_status": self.public_release_status,
            "public_release_policy": self.public_release_policy,
            "release_next_step": self.release_next_step,
            "license_evidence_url": self.license_evidence_url,
            "attribution": self.attribution,
            "version": self.version,
            "expected_download_bytes": self.expected_download_bytes,
            "expected_unpacked_bytes": self.expected_unpacked_bytes,
            "hash_policy": self.hash_policy,
        }
        return {key: value for key, value in data.items() if value is not None}


ASSET_DOSSIERS: dict[str, AssetDossier] = {
    "tinyshakespeare": AssetDossier(
        id="tinyshakespeare",
        asset_type="dataset",
        display_name="Tiny Shakespeare",
        source_url=TINY_SHAKESPEARE_URL,
        citation="Project Gutenberg eBook 100: The Complete Works of William Shakespeare.",
        license="Public domain in the United States; Project Gutenberg trademark terms apply to Project Gutenberg distributions",
        license_spdx=None,
        license_status="public-domain-us",
        terms_summary="MLPerf EDU generates the small corpus from the Project Gutenberg public-domain Shakespeare source and records the source URL and recipe.",
        public_result_use="score-bearing candidate with fetch-from-source recipe and attribution",
        public_release_status="public-ok-fetch-only",
        public_release_policy="Fetch the Project Gutenberg source and generate the MLPerf EDU tiny excerpt deterministically. Preserve the source URL, recipe, and Project Gutenberg terms link in reports.",
        release_next_step="Keep generated-corpus recipe, source URL, and hashes in public artifacts.",
        license_evidence_url="https://www.gutenberg.org/policy/license.html",
        attribution="Project Gutenberg eBook 100: The Complete Works of William Shakespeare.",
        version=TINY_SHAKESPEARE_VERSION,
        expected_download_bytes=5_600_000,
        expected_unpacked_bytes=1_115_394,
        hash_policy="Reports and provenance manifests record computed hashes from the generated tiny excerpt, train/validation split, raw Project Gutenberg source, and recipe marker when present.",
    ),
    "movielens-100k": AssetDossier(
        id="movielens-100k",
        asset_type="dataset",
        display_name="MovieLens 100K",
        source_url=MOVIELENS_100K_URL,
        citation="Harper and Konstan, The MovieLens Datasets, ACM TiiS 2015.",
        license="GroupLens dataset terms",
        license_spdx=None,
        license_status="noncommercial-research-education",
        terms_summary="Use for research and education with required GroupLens citation/acknowledgment; not an unrestricted redistributable asset.",
        public_result_use="score-bearing candidate with documented download-from-source flow",
        public_release_status="restricted-needs-approval",
        public_release_policy="Fetch from GroupLens only and do not redistribute. Public MLPerf EDU score-bearing use needs explicit approval because the terms prohibit redistribution and commercial/revenue-bearing use without permission.",
        release_next_step="Ask MLCommons reviewers whether MovieLens remains score-bearing with an official fetch-only policy, or move this workload to systems-only until permission is recorded.",
        license_evidence_url="https://files.grouplens.org/datasets/movielens/ml-100k-README.txt",
        attribution="F. Maxwell Harper and Joseph A. Konstan. The MovieLens Datasets, ACM TiiS 2015.",
        version="ml-100k",
        expected_download_bytes=4_924_029,
        expected_unpacked_bytes=16_100_896,
    ),
    "mnist": AssetDossier(
        id="mnist",
        asset_type="dataset",
        display_name="MNIST",
        source_url=MNIST_SOURCE,
        citation="LeCun et al., gradient-based learning applied to document recognition, 1998.",
        license="Creative Commons Attribution-ShareAlike 3.0",
        license_spdx="CC-BY-SA-3.0",
        license_status="cc-by-sa-3.0",
        terms_summary="Standard educational vision dataset fetched through torchvision mirrors; preserve attribution, license link, and share-alike obligations for redistributed derivatives.",
        public_result_use="score-bearing candidate with attribution and fetch-from-source flow",
        public_release_status="public-ok-with-attribution",
        public_release_policy="Fetch from upstream mirrors; reports and packages must preserve attribution and the CC BY-SA 3.0 license reference. Do not imply licensor endorsement.",
        release_next_step="Keep attribution and license metadata in report, CSV, HTML, and package artifacts.",
        license_evidence_url="https://keras.io/api/datasets/mnist/",
        attribution="Yann LeCun and Corinna Cortes; derivative of original NIST datasets.",
        version="torchvision-MNIST",
        expected_unpacked_bytes=66_544_770,
    ),
    "cifar100": AssetDossier(
        id="cifar100",
        asset_type="dataset",
        display_name="CIFAR-100",
        source_url=CIFAR100_URL,
        citation="Krizhevsky, Learning Multiple Layers of Features from Tiny Images, 2009.",
        license="citation requested; no explicit license identified in the official dataset page",
        license_spdx=None,
        license_status="source-citation-no-license",
        terms_summary="Standard educational vision dataset downloaded from the University of Toronto source; official page requests citation but does not state a permissive redistribution license.",
        public_result_use="score-bearing candidate after release review",
        public_release_status="needs-release-decision",
        public_release_policy="Fetch from the official Toronto source and avoid redistributing data in benchmark packages until terms are resolved.",
        release_next_step="Obtain explicit release terms or switch the public score-bearing vision workload to a dataset with a clear open license.",
        license_evidence_url="https://www.cs.toronto.edu/~kriz/cifar.html",
        attribution="Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.",
        version="cifar-100-python",
        expected_download_bytes=169_001_437,
        expected_unpacked_bytes=186_301_098,
    ),
    "fashion-mnist": AssetDossier(
        id="fashion-mnist",
        asset_type="dataset",
        display_name="Fashion-MNIST",
        source_url="https://github.com/zalandoresearch/fashion-mnist",
        citation="Xiao, Rasul, and Vollgraf, Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms, 2017.",
        license="MIT License",
        license_spdx="MIT",
        license_status="mit",
        terms_summary="Permissively licensed image classification dataset fetched through torchvision mirrors; preserve attribution and the MIT license reference.",
        public_result_use="score-bearing candidate with attribution and fetch-from-source flow",
        public_release_status="public-ok-with-attribution",
        public_release_policy="Fetch from upstream mirrors; reports and packages must preserve attribution and the MIT license reference.",
        release_next_step="Keep source, citation, and license metadata in report, CSV, HTML, and package artifacts.",
        license_evidence_url="https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE",
        attribution="Han Xiao, Kashif Rasul, and Roland Vollgraf; Zalando Research.",
        version="torchvision-FashionMNIST",
        expected_unpacked_bytes=31_000_000,
    ),
    "prompt-suite-local": AssetDossier(
        id="prompt-suite-local",
        asset_type="dataset",
        display_name="MLPerf EDU deterministic prompt suite",
        source_url="mlperf-edu://bundled/prompts",
        citation="Bundled deterministic prompts maintained by MLPerf EDU.",
        license="project license",
        license_spdx=None,
        license_status="bundled-project-asset",
        terms_summary="No external evaluation dataset is required; prompts are part of the benchmark harness.",
        public_result_use="performance-bearing functional check",
        public_release_status="public-ok-bundled",
        public_release_policy="Bundled deterministic prompts are project assets and can be redistributed with MLPerf EDU under the project license.",
        version="mlperf-edu-0.1",
        expected_download_bytes=0,
    ),
}


def asset_dossier(asset_id: str | None, *, declared_source: str | None = None) -> dict[str, Any]:
    if not asset_id:
        return {}
    dossier = ASSET_DOSSIERS.get(asset_id)
    if dossier:
        data = dossier.to_dict()
    else:
        data = {
            "id": asset_id,
            "type": "dataset",
            "display_name": asset_id,
            "source_url": declared_source or asset_id,
            "citation": declared_source or "",
            "license": "unknown",
            "license_status": "unknown",
            "terms_summary": "No structured asset dossier is registered yet.",
            "public_result_use": "requires review before public scoring",
            "public_release_status": "needs-release-decision",
            "public_release_policy": "Register a structured asset dossier before treating this dataset as a public MLPerf EDU result asset.",
            "release_next_step": "Add source, citation, license, release policy, size, and hash-policy metadata.",
        }
    if declared_source:
        data.setdefault("declared_source", declared_source)
    return data


def has_asset_dossier(asset_id: str | None) -> bool:
    return bool(asset_id) and asset_id in ASSET_DOSSIERS


def huggingface_model_dossier(model_source: dict[str, Any], *, model_name: str | None = None, model_id: str | None = None) -> dict[str, Any]:
    resolved_model_id = model_id or model_source.get("default_model_id") or model_name or "huggingface-model"
    source_url = f"https://huggingface.co/{resolved_model_id}" if "/" in str(resolved_model_id) else str(resolved_model_id)
    aliases = model_source.get("aliases") if isinstance(model_source.get("aliases"), dict) else {}
    license_value = str(model_source.get("license", "unknown"))
    normalized_license = license_value.lower()
    permissive = normalized_license in {"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause"}
    data = {
        "id": str(resolved_model_id),
        "type": "model",
        "display_name": model_name or str(resolved_model_id),
        "source_url": source_url,
        "provider": "Hugging Face",
        "license": license_value,
        "license_spdx": license_value if permissive else None,
        "license_status": "declared-by-upstream" if model_source.get("license") else "requires-review",
        "default_alias": model_source.get("default_alias"),
        "aliases": aliases,
        "terms_summary": "Model is fetched from its upstream Hugging Face repository; preserve upstream license and model card attribution.",
        "public_result_use": "performance-bearing candidate when the selected model license is compatible",
        "public_release_status": "public-ok-with-attribution" if permissive else "needs-release-decision",
        "public_release_policy": (
            "Fetch model weights from the upstream Hugging Face repository and preserve "
            "the model card, license, provider, and revision metadata in public artifacts."
            if permissive
            else "Resolve the upstream model license before treating this model as public-result eligible."
        ),
        "release_next_step": (
            "Keep Hugging Face model id, source URL, and license metadata in report, CSV, HTML, and package artifacts."
            if permissive
            else "Select a permissive model or record an explicit MLCommons-approved model policy."
        ),
    }
    for key in ("selection_rationale", "size_rationale", "backend_rationale"):
        if model_source.get(key):
            data[key] = model_source[key]

    alias_rationales = model_source.get("alias_rationales")
    if isinstance(alias_rationales, dict):
        data["alias_rationales"] = alias_rationales
        rationale = model_rationale_for(resolved_model_id, aliases, alias_rationales)
        if rationale:
            data["selected_model_rationale"] = rationale
    return data


def model_rationale_for(model_id: str, aliases: dict[str, Any], alias_rationales: dict[str, Any]) -> str | None:
    candidates = [str(model_id)]
    for alias, target in aliases.items():
        if str(target) == str(model_id):
            candidates.append(str(alias))
    for candidate in candidates:
        if candidate in alias_rationales:
            return str(alias_rationales[candidate])
    return None


def data_root() -> Path:
    override = os.environ.get("MLPERF_EDU_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    from .registry import find_project_root

    return find_project_root() / "datasets" / "local_tensors"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_cifar100_dataset(*, root: Path, train: bool, download: bool, transform=None):
    from torchvision.datasets import CIFAR100

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=CIFAR100_PICKLE_WARNING,
            category=Warning,
        )
        return CIFAR100(root=str(root), train=train, download=download, transform=transform)


def load_fashion_mnist_dataset(*, root: Path, train: bool, download: bool, transform=None):
    from torchvision.datasets import FashionMNIST

    return FashionMNIST(root=str(root), train=train, download=download, transform=transform)


def tinyshakespeare_paths(root: Path | None = None) -> dict[str, Path]:
    base = (root or data_root()).resolve()
    return {
        "root": base,
        "raw": base / "tinyshakespeare_gutenberg_raw.txt",
        "full": base / "tinyshakespeare.txt",
        "train": base / "tinyshakespeare_train.txt",
        "val": base / "tinyshakespeare_val.txt",
        "recipe": base / "tinyshakespeare_recipe.txt",
    }


def movielens_paths(root: Path | None = None) -> dict[str, Path]:
    if root is not None:
        base = root.resolve()
    else:
        override = os.environ.get("MLPERF_EDU_DATA_DIR")
        from .registry import find_project_root

        base = (Path(override).expanduser().resolve() / "movielens") if override else find_project_root() / "data" / "movielens"
    dataset = base / "ml-100k"
    return {
        "root": base,
        "dataset": dataset,
        "zip": base / "ml-100k.zip",
        "ratings": dataset / "u.data",
        "users": dataset / "u.user",
        "items": dataset / "u.item",
    }


def mnist_paths(root: Path | None = None) -> dict[str, Path]:
    if root is not None:
        base = root.resolve()
    else:
        override = os.environ.get("MLPERF_EDU_DATA_DIR")
        from .registry import find_project_root

        base = (Path(override).expanduser().resolve() / "mnist") if override else find_project_root() / "data" / "mnist"
    return {
        "root": base,
        "raw": base / "MNIST" / "raw",
        "processed": base / "MNIST" / "processed",
    }


def cifar100_paths(root: Path | None = None) -> dict[str, Path]:
    if root is not None:
        base = root.resolve()
    else:
        override = os.environ.get("MLPERF_EDU_DATA_DIR")
        from .registry import find_project_root

        base = (Path(override).expanduser().resolve() / "cifar100") if override else find_project_root() / "data" / "cifar100"
    return {
        "root": base,
        "dataset": base / "cifar-100-python",
        "tar": base / "cifar-100-python.tar.gz",
    }


def fashion_mnist_paths(root: Path | None = None) -> dict[str, Path]:
    if root is not None:
        base = root.resolve()
    else:
        override = os.environ.get("MLPERF_EDU_DATA_DIR")
        from .registry import find_project_root

        base = (Path(override).expanduser().resolve() / "fashion-mnist") if override else find_project_root() / "data" / "fashion-mnist"
    return {
        "root": base,
        "raw": base / "FashionMNIST" / "raw",
        "processed": base / "FashionMNIST" / "processed",
    }


def ensure_tinyshakespeare(*, download: bool = True, root: Path | None = None) -> DatasetAsset:
    paths = tinyshakespeare_paths(root)
    base = paths["root"]
    raw = paths["raw"]
    full = paths["full"]
    train = paths["train"]
    val = paths["val"]
    recipe = paths["recipe"]
    base.mkdir(parents=True, exist_ok=True)
    recipe_text = (
        f"source_url={TINY_SHAKESPEARE_URL}\n"
        f"version={TINY_SHAKESPEARE_VERSION}\n"
        f"target_chars={TINY_SHAKESPEARE_TARGET_CHARS}\n"
        "normalization=strip Project Gutenberg header/footer, normalize line endings, "
        "truncate to target_chars\n"
    )

    known_old_cache = full.exists() and sha256_file(full) == KARPATHY_TINY_SHAKESPEARE_SHA256
    recipe_mismatch = recipe.exists() and recipe.read_text(encoding="utf-8") != recipe_text
    needs_generated_source = not full.exists() or known_old_cache or recipe_mismatch
    if needs_generated_source:
        if not download:
            raise FileNotFoundError(
                f"TinyShakespeare is missing at {full}. "
                "Run `mlperf fetch --workload nanogpt-train --profile max`."
            )
        tmp = raw.with_suffix(".download")
        _download(TINY_SHAKESPEARE_URL, tmp)
        tmp.replace(raw)
        generated = _generate_tinyshakespeare_from_gutenberg(raw.read_text(encoding="utf-8-sig"))
        full.write_text(generated, encoding="utf-8")
        recipe.write_text(recipe_text, encoding="utf-8")

    if needs_generated_source or not train.exists() or not val.exists():
        text = full.read_text(encoding="utf-8")
        split_idx = int(len(text) * 0.8)
        train.write_text(text[:split_idx], encoding="utf-8")
        val.write_text(text[split_idx:], encoding="utf-8")

    files = [full, train, val]
    for optional in (raw, recipe):
        if optional.exists():
            files.append(optional)
    digest = hashlib.sha256()
    n_bytes = 0
    for path in files:
        n_bytes += path.stat().st_size
        digest.update(str(path.relative_to(base)).encode("utf-8") + b"\0")
        digest.update(sha256_file(path).encode("ascii") + b"\n")
    return DatasetAsset(
        name="tinyshakespeare",
        root=base,
        files=tuple(files),
        sha256=f"sha256:{digest.hexdigest()}",
        n_bytes=n_bytes,
        source=TINY_SHAKESPEARE_URL,
    )


def _generate_tinyshakespeare_from_gutenberg(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    start = normalized.find(start_marker)
    if start != -1:
        line_end = normalized.find("\n", start)
        normalized = normalized[line_end + 1 :] if line_end != -1 else normalized[start + len(start_marker) :]
    end = normalized.find(end_marker)
    if end != -1:
        normalized = normalized[:end]
    normalized = normalized.strip() + "\n"
    return normalized[:TINY_SHAKESPEARE_TARGET_CHARS]


def ensure_cifar100(*, download: bool = True, root: Path | None = None) -> DatasetAsset:
    paths = cifar100_paths(root)
    base = paths["root"]
    dataset = paths["dataset"]
    tar_path = paths["tar"]
    base.mkdir(parents=True, exist_ok=True)

    try:
        load_cifar100_dataset(root=base, train=True, download=download)
        load_cifar100_dataset(root=base, train=False, download=download)
    except Exception as exc:
        if not download:
            raise FileNotFoundError(
                f"CIFAR-100 is missing at {base}. "
                "Run `mlperf fetch --workload resnet18-train --profile max`."
            ) from exc
        if not tar_path.exists():
            tmp = tar_path.with_suffix(".download")
            _download(CIFAR100_URL, tmp)
            tmp.replace(tar_path)
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(base)
        load_cifar100_dataset(root=base, train=True, download=False)
        load_cifar100_dataset(root=base, train=False, download=False)

    files = tuple(sorted(path for path in dataset.rglob("*") if path.is_file()))
    if not files:
        raise FileNotFoundError(f"CIFAR-100 produced no files under {dataset}")
    n_bytes = sum(path.stat().st_size for path in files)
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(base)).encode("utf-8") + b"\0")
        digest.update(sha256_file(path).encode("ascii") + b"\n")

    return DatasetAsset(
        name="cifar100",
        root=base,
        files=files,
        sha256=f"sha256:{digest.hexdigest()}",
        n_bytes=n_bytes,
        source=CIFAR100_URL,
    )


def ensure_fashion_mnist(*, download: bool = True, root: Path | None = None) -> DatasetAsset:
    paths = fashion_mnist_paths(root)
    base = paths["root"]
    base.mkdir(parents=True, exist_ok=True)

    try:
        load_fashion_mnist_dataset(root=base, train=True, download=download)
        load_fashion_mnist_dataset(root=base, train=False, download=download)
    except Exception as exc:
        if not download:
            raise FileNotFoundError(
                f"Fashion-MNIST is missing at {base}. "
                "Run `mlperf fetch --workload resnet18-train --profile max`."
            ) from exc
        raise

    files = tuple(sorted(path for path in (base / "FashionMNIST").rglob("*") if path.is_file()))
    if not files:
        raise FileNotFoundError(f"Fashion-MNIST produced no files under {base}")
    n_bytes = sum(path.stat().st_size for path in files)
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(base)).encode("utf-8") + b"\0")
        digest.update(sha256_file(path).encode("ascii") + b"\n")

    return DatasetAsset(
        name="fashion-mnist",
        root=base,
        files=files,
        sha256=f"sha256:{digest.hexdigest()}",
        n_bytes=n_bytes,
        source=FASHION_MNIST_SOURCE,
    )


def ensure_mnist(*, download: bool = True, root: Path | None = None) -> DatasetAsset:
    paths = mnist_paths(root)
    base = paths["root"]
    base.mkdir(parents=True, exist_ok=True)
    from torchvision.datasets import MNIST

    try:
        MNIST(root=str(base), train=True, download=download)
        MNIST(root=str(base), train=False, download=download)
    except Exception as exc:
        if not download:
            raise FileNotFoundError(
                f"MNIST is missing at {base}. "
                "Run `mlperf fetch --workload anomaly-ae-train --profile max`."
            ) from exc
        _download_mnist_with_curl_fallback(base)
        MNIST(root=str(base), train=True, download=False)
        MNIST(root=str(base), train=False, download=False)

    files = tuple(sorted(path for path in (base / "MNIST").rglob("*") if path.is_file()))
    if not files:
        raise FileNotFoundError(f"MNIST produced no files under {base}")

    n_bytes = sum(path.stat().st_size for path in files)
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(base)).encode("utf-8") + b"\0")
        digest.update(sha256_file(path).encode("ascii") + b"\n")

    return DatasetAsset(
        name="mnist",
        root=base,
        files=files,
        sha256=f"sha256:{digest.hexdigest()}",
        n_bytes=n_bytes,
        source=MNIST_SOURCE,
    )


def _download_mnist_with_curl_fallback(base: Path) -> None:
    raw = base / "MNIST" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for filename, _md5 in (
        ("train-images-idx3-ubyte.gz", ""),
        ("train-labels-idx1-ubyte.gz", ""),
        ("t10k-images-idx3-ubyte.gz", ""),
        ("t10k-labels-idx1-ubyte.gz", ""),
    ):
        gz_path = raw / filename
        extracted_path = raw / filename.removesuffix(".gz")
        if extracted_path.exists():
            continue
        _download(f"https://ossci-datasets.s3.amazonaws.com/mnist/{filename}", gz_path)
        with gzip.open(gz_path, "rb") as src, extracted_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)


def ensure_movielens_100k(*, download: bool = True, root: Path | None = None) -> DatasetAsset:
    paths = movielens_paths(root)
    base = paths["root"]
    dataset = paths["dataset"]
    zip_path = paths["zip"]
    base.mkdir(parents=True, exist_ok=True)

    required = (paths["ratings"], paths["users"], paths["items"])
    if not all(path.exists() for path in required):
        if not download:
            raise FileNotFoundError(
                f"MovieLens-100K is missing at {dataset}. "
                "Run `mlperf fetch --workload micro-dlrm-train --profile max`."
            )
        if not zip_path.exists():
            tmp = zip_path.with_suffix(".download")
            _download(MOVIELENS_100K_URL, tmp)
            tmp.replace(zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(base)

    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"MovieLens-100K is incomplete; missing: {missing}")

    files = tuple(sorted(path for path in dataset.iterdir() if path.is_file()))
    n_bytes = sum(path.stat().st_size for path in files)
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8") + b"\0")
        digest.update(sha256_file(path).encode("ascii") + b"\n")

    return DatasetAsset(
        name="movielens-100k",
        root=dataset,
        files=files,
        sha256=f"sha256:{digest.hexdigest()}",
        n_bytes=n_bytes,
        source=MOVIELENS_100K_URL,
    )


def _download(url: str, destination: Path) -> None:
    # Download to a process- and call-unique path first, then atomically move
    # it into place. Two concurrent fetches of the same asset (e.g. two
    # workload runs racing to populate a shared dataset cache) must not write
    # through the same temp file, or a slower writer's in-progress download
    # can be clobbered mid-write by the other process before either renames.
    unique_tmp = destination.with_name(f"{destination.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.part")
    try:
        try:
            urllib.request.urlretrieve(url, unique_tmp)
        except Exception as urllib_exc:
            if shutil.which("curl"):
                try:
                    subprocess.run(
                        ["curl", "--fail", "--location", "--silent", "--show-error", url, "--output", str(unique_tmp)],
                        check=True,
                    )
                except subprocess.CalledProcessError as curl_exc:
                    raise RuntimeError(f"failed to download {url} with urllib and curl") from curl_exc
            else:
                raise RuntimeError(f"failed to download {url}") from urllib_exc
        unique_tmp.replace(destination)
    finally:
        if unique_tmp.exists():
            unique_tmp.unlink()
