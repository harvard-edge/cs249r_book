from __future__ import annotations

import csv
import hashlib
import io
import os
import shutil
import subprocess
import tarfile
import zipfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TINY_SHAKESPEARE_UPSTREAM_COMMIT = "6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e"
TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    f"{TINY_SHAKESPEARE_UPSTREAM_COMMIT}/data/tinyshakespeare/input.txt"
)
TINY_SHAKESPEARE_VERSION = f"karpathy-char-rnn-{TINY_SHAKESPEARE_UPSTREAM_COMMIT}"
TINY_SHAKESPEARE_TARGET_CHARS = 1_115_394
KARPATHY_TINY_SHAKESPEARE_SHA256 = (
    "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
)
FASHION_MNIST_SOURCE = "torchvision://FashionMNIST"
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_HF_REPO_ID = "uoft-cs/cifar10"
CIFAR10_HF_REVISION = "0b2714987fa478483af9968de7c934580d0bb9a2"
CIFAR10_HF_FILES = {
    "plain_text/test-00000-of-00001.parquet": "841389e6f2d64f28bf17310e430aebac20ec3ba611a3c5e231dc93c645ce84de",
}
MLPERF_TINY_COMMIT = "1afd2c9820f795965a6134facd0b4dfae41ef23f"
MLPERF_TINY_IMAGE_BASE_URL = (
    "https://raw.githubusercontent.com/mlcommons/tiny/"
    f"{MLPERF_TINY_COMMIT}/benchmark/training/image_classification"
)
MLPERF_TINY_IMAGE_FLOAT_MODEL_SHA256 = (
    "b5c0046d6e0328b4956afd6baa29555a29b1f1c65bdd45aaed75b7cd484d9f79"
)
MLPERF_TINY_IMAGE_PERF_INDICES_SHA256 = (
    "3bd4a88eeb4c50fad652d0f24c8af13bc9219ba2878aea47c6536bfbeb43024d"
)
EEMBC_RUNNER_COMMIT = "cf7c2f2634608a7c0ea7458ab7cb3379f2863424"
EEMBC_RUNNER_ARCHIVE_URL = (
    f"https://github.com/eembc/benchmark-runner-ml/archive/{EEMBC_RUNNER_COMMIT}.tar.gz"
)
EEMBC_RUNNER_ARCHIVE_SHA256 = (
    "87e431a6b4d3f011d672180a3fb1f08856d8074310f37653d5388ec2affc5209"
)
MLPERF_TINY_KWS_MODEL_BASE_URL = (
    "https://raw.githubusercontent.com/mlcommons/tiny/"
    f"{MLPERF_TINY_COMMIT}/benchmark/training/keyword_spotting/trained_models"
)
MLPERF_TINY_KWS_FLOAT_MODEL_SHA256 = (
    "e5004c6f1012246e33fa068d8488325538e0444073cd361f5a7edb40c73f12d2"
)
MLPERF_TINY_KWS_INT8_MODEL_SHA256 = (
    "aeea436800704fce17b17292e4412630ad856e9d777c044c64ef748a880bd0ae"
)
MLPERF_TINY_VWW_COMMIT = "4addd0fa08d216e20637637874e084895f289da4"
MLPERF_TINY_VWW_MODEL_BASE_URL = (
    "https://raw.githubusercontent.com/mlcommons/tiny/"
    f"{MLPERF_TINY_VWW_COMMIT}/benchmark/training/visual_wake_words/trained_models"
)
MLPERF_TINY_VWW_FLOAT_MODEL_SHA256 = (
    "115bbc094d2119561320a21f01b6500a18bea8cc8589282ab007097bec8af38c"
)
MLPERF_TINY_VWW_INT8_MODEL_SHA256 = (
    "597a384c8c2c8a1276f04702f25013b7838f2f814f1ca7c174d295b73e3d6b7b"
)
MLPERF_TINY_VWW_ARCHIVE_URL = (
    "https://www.silabs.com/public/files/github/machine_learning/benchmarks/"
    "datasets/vw_coco2014_96.tar.gz"
)
MLPERF_TINY_VWW_ARCHIVE_SHA256 = (
    "f8746b9e44f8a7a4293f73be9ba6e8da9239fe69798d42364aae62b915cfab58"
)
MLPERF_TINY_VWW_ARCHIVE_BYTES = 234_810_765
MLPERF_TINY_VWW_LABELS_SHA256 = (
    "3697ca57c48b23b21602ae9bdb32b1925407a1d41d79167cdfb365054cb9c33d"
)
MLPERF_TINY_VWW_DATASET_SHA256 = (
    "8de5c9f84131c5a77e807356362865e9471b6ab6fc2411db0e7a0c5e129eb3b3"
)
MLPERF_TINY_VWW_DATASET_BYTES = 2_747_212
GLUE_SST2_URL = "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"
GLUE_SST2_ZIP_SHA256 = (
    "d67e16fb55739c1b32cdce9877596db1c127dc322d93c082281f64057c16deaa"
)
OGBN_ARXIV_URL = "https://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip"
OGBN_ARXIV_ZIP_SHA256 = (
    "49f85c801589ecdcc52cfaca99693aaea7b8af16a9ac3f41dd85a5f3193fe276"
)
ETT_DATASET_COMMIT = "1d16c8f4f943005d613b5bc962e9eeb06058cf07"
ETTM1_URL = (
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/"
    f"{ETT_DATASET_COMMIT}/ETT-small/ETTm1.csv"
)
ETTM1_SHA256 = "6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e"
NANOBEIR_REPO_ID = "sentence-transformers/NanoBEIR-en"
NANOBEIR_REVISION = "beb106fbcfaa599c508c667041bf8c85fd78736b"
NANOBEIR_RERANKING_FILES = {
    "bm25/NanoMSMARCO-00000-of-00001.parquet": "8496f6787768fc06558cc40debe66ac7cb964ff0b6304ef5c4302923b5ef4225",
    "corpus/NanoMSMARCO-00000-of-00001.parquet": "685715c7e0a66d0219572dcd43c3905782868d1aae885259768431f7d7eda830",
    "qrels/NanoMSMARCO-00000-of-00001.parquet": "6cd84c97a6ed813ffccbbb0b7aacc3051641f40a5869e0a15415823caf65c0d1",
    "queries/NanoMSMARCO-00000-of-00001.parquet": "7cb9d7534660847f303211b9bdf84bcb3a3530f6e20e3c6050e77fc7ae77d0cd",
    "bm25/NanoNFCorpus-00000-of-00001.parquet": "e4f7bdebf7e25fe2d2f1ea172cb1415315704bdc94900dc095e8d9df52113cda",
    "corpus/NanoNFCorpus-00000-of-00001.parquet": "d50e7ac973d4367434b68c1e7eb54d7827b29d85aa54a1dde42883f05fbf7d95",
    "qrels/NanoNFCorpus-00000-of-00001.parquet": "d97ea8176db52aa04773f2459d02e20c582ed9aa694801201cd21e841a00f200",
    "queries/NanoNFCorpus-00000-of-00001.parquet": "e9a58c2e1f392a83b26eade3d9838f7448c8a6cdb34f7257f3475cb76024aec2",
    "bm25/NanoNQ-00000-of-00001.parquet": "da76a9518c49afb494cf7085861ca6f303acca823fd2287da1fcd1f747ddbafd",
    "corpus/NanoNQ-00000-of-00001.parquet": "85d306945cd09cb748ca5b198b281a4f1f034b8240f8c5ecacceb68e38a1db0a",
    "qrels/NanoNQ-00000-of-00001.parquet": "f08f73ba0246a9ec1282ca26b48faa24cffb7e1e223354b7fb14fa9f4339e112",
    "queries/NanoNQ-00000-of-00001.parquet": "3731f4ac7be9dc1054783ea700ee883d8fd8ad2283da259b1216fff0b4107a5e",
}


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
        citation="Karpathy char-rnn Tiny Shakespeare corpus; source text derived from public-domain Shakespeare works.",
        license="upstream char-rnn repository is MIT; underlying Shakespeare text is public domain in the United States",
        license_spdx=None,
        license_status="mit-repository-public-domain-text",
        terms_summary="The exact 1,115,394-character corpus and 90/10 split are inherited from nanoGPT's pinned Shakespeare character-data recipe.",
        public_result_use="pinned nanoGPT score-bearing candidate with fetch-from-source recipe and attribution",
        public_release_status="public-ok-fetch-only",
        public_release_policy="Fetch the exact corpus from the pinned char-rnn commit. Preserve repository attribution, the 90/10 split recipe, and content hashes.",
        release_next_step="Keep the pinned commit, source URL, split recipe, and hashes in public artifacts.",
        license_evidence_url="https://github.com/karpathy/char-rnn/blob/master/LICENSE",
        attribution="Andrej Karpathy char-rnn and nanoGPT; William Shakespeare source text.",
        version=TINY_SHAKESPEARE_VERSION,
        expected_download_bytes=5_600_000,
        expected_unpacked_bytes=1_115_394,
        hash_policy="Reports and provenance manifests record the pinned upstream corpus hash, exact 90/10 train/validation split hashes, and recipe marker.",
    ),
    "cifar10": AssetDossier(
        id="cifar10",
        asset_type="dataset",
        display_name="CIFAR-10",
        source_url=f"https://huggingface.co/datasets/{CIFAR10_HF_REPO_ID}/tree/{CIFAR10_HF_REVISION}",
        citation="Krizhevsky, Learning Multiple Layers of Features from Tiny Images, 2009.",
        license="citation requested; no explicit license identified on the official dataset page",
        license_spdx=None,
        license_status="source-citation-no-license",
        terms_summary="MLPerf Tiny uses CIFAR-10 for image classification. MLPerf EDU fetches only the pinned test Parquet required by the official 200-sample accuracy set and does not package the data.",
        public_result_use="MLPerf Tiny-derived score-bearing candidate after release review",
        public_release_status="needs-release-decision",
        public_release_policy="Fetch from the official Toronto source and avoid redistributing the dataset in benchmark packages until release terms are resolved.",
        release_next_step="Record the MLCommons release decision for fetch-only public benchmark use.",
        license_evidence_url="https://www.cs.toronto.edu/~kriz/cifar.html",
        attribution="Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.",
        version="cifar-10-python",
        expected_download_bytes=23_940_850,
        expected_unpacked_bytes=23_940_850,
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
        public_result_use="standalone educational lab asset; never a benchmark result",
        public_release_status="public-ok-with-attribution",
        public_release_policy="Fetch from upstream mirrors; reports and packages must preserve attribution and the MIT license reference.",
        release_next_step="Keep source, citation, and license metadata with the standalone lab.",
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
        citation="Versioned deterministic prompts maintained by MLPerf EDU contributors.",
        license="CC0-1.0",
        license_spdx="CC0-1.0",
        license_status="bundled-project-asset",
        terms_summary="No external evaluation dataset is required. The SLM continuation fixture and deterministic NanoGPT token-prompt recipe are project-authored under CC0-1.0; generated prompts are bound by SHA-256.",
        public_result_use="performance-bearing functional check",
        public_release_status="public-ok-bundled",
        public_release_policy="Redistribute the attributed CC0 SLM fixture and preserve the fixed-seed recipe and SHA-256 identity for generated NanoGPT token prompts.",
        version="mlperf-edu-prompt-assets-0.2",
        expected_download_bytes=0,
    ),
    "mlperf-tiny-kws-eval": AssetDossier(
        id="mlperf-tiny-kws-eval",
        asset_type="dataset",
        display_name="MLPerf Tiny keyword-spotting accuracy set",
        source_url=EEMBC_RUNNER_ARCHIVE_URL,
        citation="MLCommons MLPerf Tiny keyword spotting; Google Speech Commands v2 (Warden, 2018).",
        license="dataset access and redistribution remain subject to the MLCommons/EEMBC terms and Speech Commands CC BY 4.0 attribution",
        license_spdx=None,
        license_status="mlcommons-review-required",
        terms_summary="The pinned EEMBC runner repository contains 1,000 preprocessed 49 by 10 INT8 MFCC examples and labels used by the MLPerf Tiny accuracy contract.",
        public_result_use="MLPerf Tiny-derived performance candidate with a fixed 90% quality gate",
        public_release_status="needs-release-decision",
        public_release_policy="Fetch the pinned repository at run time. Do not package or republish the evaluation examples until MLCommons confirms the release policy.",
        release_next_step="Record MLCommons approval for public fetch-only use before release promotion.",
        license_evidence_url="https://github.com/eembc/benchmark-runner-ml/blob/main/README.md",
        attribution="MLCommons, EEMBC, and Pete Warden/Google Speech Commands contributors.",
        version=f"eembc-runner-{EEMBC_RUNNER_COMMIT}",
        expected_download_bytes=2_183_000,
        expected_unpacked_bytes=4_100_000,
    ),
    "mlperf-tiny-vww-eval": AssetDossier(
        id="mlperf-tiny-vww-eval",
        asset_type="dataset",
        display_name="MLPerf Tiny visual-wake-words accuracy set",
        source_url=MLPERF_TINY_VWW_ARCHIVE_URL,
        citation="MLCommons MLPerf Tiny visual wake words; COCO 2014 and the Visual Wake Words dataset.",
        license="COCO image licenses and MLCommons/EEMBC accuracy-set terms require release review",
        license_spdx=None,
        license_status="mlcommons-coco-review-required",
        terms_summary="The pinned Silicon Labs archive supplies the 96 by 96 COCO-derived images, and the pinned EEMBC index selects the balanced 1,000-example MLPerf Tiny accuracy set.",
        public_result_use="MLPerf Tiny-derived score-bearing candidate with a fixed 80% top-1 quality gate",
        public_release_status="needs-release-decision",
        public_release_policy="Fetch the source archive at run time and do not package or republish the evaluation images until MLCommons confirms the release policy.",
        release_next_step="Record MLCommons approval for public fetch-only use before release promotion.",
        license_evidence_url="https://cocodataset.org/#termsofuse",
        attribution="MLCommons, EEMBC, Silicon Labs, COCO, and Visual Wake Words contributors.",
        version=(
            f"mlcommons-tiny-{MLPERF_TINY_VWW_COMMIT}-eembc-{EEMBC_RUNNER_COMMIT}"
        ),
        expected_download_bytes=MLPERF_TINY_VWW_ARCHIVE_BYTES,
        expected_unpacked_bytes=MLPERF_TINY_VWW_DATASET_BYTES,
    ),
    "sst2": AssetDossier(
        id="sst2",
        asset_type="dataset",
        display_name="GLUE SST-2",
        source_url=GLUE_SST2_URL,
        citation="Socher et al., Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank, 2013; GLUE SST-2 packaging.",
        license="dataset terms require review; the GLUE archive does not supply a single permissive redistribution license",
        license_spdx=None,
        license_status="source-citation-no-license",
        terms_summary="MLPerf EDU fetches the official GLUE SST-2 archive and evaluates only the labeled 872-example development split.",
        public_result_use="pinned DistilBERT text-classification performance candidate",
        public_release_status="needs-release-decision",
        public_release_policy="Fetch from the official GLUE host and do not package the corpus until release terms are confirmed.",
        release_next_step="Record the release decision for public fetch-only benchmark use.",
        license_evidence_url="https://gluebenchmark.com/",
        attribution="Richard Socher and Stanford NLP; GLUE benchmark maintainers.",
        version="GLUE-SST-2-2018",
        expected_download_bytes=7_438_000,
        expected_unpacked_bytes=24_500_000,
    ),
    "ogbn-arxiv": AssetDossier(
        id="ogbn-arxiv",
        asset_type="dataset",
        display_name="OGB ogbn-arxiv",
        source_url=OGBN_ARXIV_URL,
        citation="Hu et al., Open Graph Benchmark: Datasets for Machine Learning on Graphs, NeurIPS 2020.",
        license="Open Graph Benchmark dataset terms; Microsoft Academic Graph-derived records require attribution and release review",
        license_spdx=None,
        license_status="ogb-terms-review-required",
        terms_summary="The official OGB loader supplies 169,343 papers, 1,166,243 citation edges, 128-dimensional features, 40 classes, and the time-based split.",
        public_result_use="official OGB node-classification performance candidate",
        public_release_status="needs-release-decision",
        public_release_policy="Fetch the pinned official archive and do not redistribute it in benchmark packages until OGB/MAG terms are confirmed.",
        release_next_step="Record the MLCommons release decision for fetch-only OGB use.",
        license_evidence_url="https://ogb.stanford.edu/docs/nodeprop/",
        attribution="Open Graph Benchmark team and Microsoft Academic Graph.",
        version="ogbn-arxiv-v1",
        expected_download_bytes=83_058_288,
        expected_unpacked_bytes=83_201_248,
    ),
    "ettm1": AssetDossier(
        id="ettm1",
        asset_type="dataset",
        display_name="Electricity Transformer Temperature, minute-level split 1",
        source_url=ETTM1_URL,
        citation="Zhou et al., Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting, AAAI 2021.",
        license="ETDataset repository is GPL-3.0; dataset-specific redistribution terms require release review",
        license_spdx=None,
        license_status="source-license-release-review-required",
        terms_summary="PatchTST evaluates the pinned ETTm1 CSV using the official 12-month training, four-month validation, and four-month test boundaries with train-split standardization.",
        public_result_use="official PatchTST time-series forecasting candidate",
        public_release_status="needs-release-decision",
        public_release_policy="Fetch the exact CSV from the pinned ETTDataset commit and do not package the data until MLCommons records a release decision.",
        release_next_step="Record the MLCommons release decision for fetch-only ETTm1 use.",
        license_evidence_url="https://github.com/zhouhaoyi/ETDataset/blob/main/LICENSE",
        attribution="Haoyi Zhou and ETTDataset contributors; PatchTST authors.",
        version=f"ETTm1-{ETT_DATASET_COMMIT}",
        expected_download_bytes=10_000_000,
        expected_unpacked_bytes=10_000_000,
    ),
    "nanobeir-reranking": AssetDossier(
        id="nanobeir-reranking",
        asset_type="dataset",
        display_name="NanoBEIR English reranking subset",
        source_url=f"https://huggingface.co/datasets/{NANOBEIR_REPO_ID}/tree/{NANOBEIR_REVISION}",
        citation="Thakur et al., BEIR, NeurIPS 2021; Sentence Transformers NanoBEIR packaging.",
        license="Apache-2.0 dataset repository metadata; component source datasets retain their original licenses",
        license_spdx=None,
        license_status="component-licenses-release-review-required",
        terms_summary="The pinned bundle contains the official BM25 top-100 rankings, corpora, queries, and relevance judgments for NanoMSMARCO, NanoNFCorpus, and NanoNQ.",
        public_result_use="official Sentence Transformers cross-encoder NanoBEIR candidate",
        public_release_status="needs-release-decision",
        public_release_policy="Fetch the twelve pinned Parquet files at run time; do not package or redistribute component datasets until their source terms are reviewed.",
        release_next_step="Record the MLCommons release decision for fetch-only NanoBEIR use and preserve component-dataset attribution.",
        license_evidence_url=f"https://huggingface.co/datasets/{NANOBEIR_REPO_ID}",
        attribution="Sentence Transformers, BEIR, MS MARCO, NFCorpus, and Natural Questions contributors.",
        version=f"NanoBEIR-en-{NANOBEIR_REVISION}",
        expected_download_bytes=6_000_000,
        expected_unpacked_bytes=6_000_000,
    ),
}


def asset_dossier(
    asset_id: str | None, *, declared_source: str | None = None
) -> dict[str, Any]:
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


def huggingface_model_dossier(
    model_source: dict[str, Any],
    *,
    model_name: str | None = None,
    model_id: str | None = None,
) -> dict[str, Any]:
    resolved_model_id = (
        model_id or model_source.get("repo_id") or model_name or "huggingface-model"
    )
    source_url = (
        f"https://huggingface.co/{resolved_model_id}"
        if "/" in str(resolved_model_id)
        else str(resolved_model_id)
    )
    license_value = str(model_source.get("license", "unknown"))
    normalized_license = license_value.lower()
    permissive = normalized_license in {
        "apache-2.0",
        "mit",
        "bsd-2-clause",
        "bsd-3-clause",
    }
    data = {
        "id": str(resolved_model_id),
        "type": "model",
        "display_name": model_name or str(resolved_model_id),
        "source_url": source_url,
        "provider": "Hugging Face",
        "license": license_value,
        "license_spdx": license_value if permissive else None,
        "license_status": "declared-by-upstream"
        if model_source.get("license")
        else "requires-review",
        "revision": model_source.get("revision"),
        "terms_summary": "Model is fetched from its upstream Hugging Face repository; preserve upstream license and model card attribution.",
        "public_result_use": "performance-bearing candidate when the selected model license is compatible",
        "public_release_status": "public-ok-with-attribution"
        if permissive
        else "needs-release-decision",
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

    return data


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


def load_cifar10_dataset(*, root: Path, train: bool, download: bool, transform=None):
    parquet = (
        root
        / "plain_text"
        / ("train-00000-of-00001.parquet" if train else "test-00000-of-00001.parquet")
    )
    if parquet.is_file():
        return CIFAR10ParquetDataset(parquet, transform=transform)
    from torchvision.datasets import CIFAR10

    return CIFAR10(root=str(root), train=train, download=download, transform=transform)


class CIFAR10ParquetDataset:
    """Torchvision-compatible view of the pinned UofT CIFAR-10 Parquet mirror."""

    def __init__(self, parquet: Path, *, transform=None):
        import pandas as pd

        self.frame = pd.read_parquet(parquet, columns=["img", "label"])
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        from PIL import Image

        row = self.frame.iloc[index]
        payload = row["img"]
        if isinstance(payload, dict):
            payload = payload.get("bytes")
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError("CIFAR-10 Parquet image is not encoded bytes")
        with Image.open(io.BytesIO(bytes(payload))) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label"])


def load_fashion_mnist_dataset(
    *, root: Path, train: bool, download: bool, transform=None
):
    from torchvision.datasets import FashionMNIST

    return FashionMNIST(
        root=str(root), train=train, download=download, transform=transform
    )


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


def cifar10_paths(root: Path | None = None) -> dict[str, Path]:
    if root is not None:
        base = root.resolve()
    else:
        override = os.environ.get("MLPERF_EDU_DATA_DIR")
        from .registry import find_project_root

        base = (
            (Path(override).expanduser().resolve() / "cifar10")
            if override
            else find_project_root() / "data" / "cifar10"
        )
    return {
        "root": base,
        "dataset": base / "cifar-10-batches-py",
        "tar": base / "cifar-10-python.tar.gz",
    }


def fashion_mnist_paths(root: Path | None = None) -> dict[str, Path]:
    if root is not None:
        base = root.resolve()
    else:
        override = os.environ.get("MLPERF_EDU_DATA_DIR")
        from .registry import find_project_root

        base = (
            (Path(override).expanduser().resolve() / "fashion-mnist")
            if override
            else find_project_root() / "data" / "fashion-mnist"
        )
    return {
        "root": base,
        "raw": base / "FashionMNIST" / "raw",
        "processed": base / "FashionMNIST" / "processed",
    }


def mlperf_tiny_kws_paths(root: Path | None = None) -> dict[str, Path]:
    if root is not None:
        base = root.resolve()
    else:
        override = os.environ.get("MLPERF_EDU_DATA_DIR")
        from .registry import find_project_root

        base = (
            (Path(override).expanduser().resolve() / "mlperf-tiny-kws")
            if override
            else find_project_root() / "data" / "mlperf-tiny-kws"
        )
    return {
        "root": base,
        "dataset": base / "kws01",
        "archive": base / f"eembc-runner-{EEMBC_RUNNER_COMMIT}.tar.gz",
        "float_model": base / "kws_ref_model_float32.tflite",
        "int8_model": base / "kws_ref_model.tflite",
    }


def mlperf_tiny_vww_paths(root: Path | None = None) -> dict[str, Path]:
    if root is not None:
        base = root.resolve()
    else:
        override = os.environ.get("MLPERF_EDU_DATA_DIR")
        from .registry import find_project_root

        base = (
            (Path(override).expanduser().resolve() / "mlperf-tiny-vww")
            if override
            else find_project_root() / "data" / "mlperf-tiny-vww"
        )
    return {
        "root": base,
        "dataset": base / "vww01",
        "images": base / "vww01" / "images",
        "labels": base / "vww01" / "y_labels.csv",
        "source_archive": base / "vw_coco2014_96.tar.gz",
        "runner_archive": base / f"eembc-runner-{EEMBC_RUNNER_COMMIT}.tar.gz",
        "float_model": base / "vww_96_float.tflite",
        "int8_model": base / "vww_96_int8.tflite",
    }


def mlperf_tiny_image_paths(root: Path | None = None) -> dict[str, Path]:
    if root is not None:
        base = root.resolve()
    else:
        override = os.environ.get("MLPERF_EDU_DATA_DIR")
        from .registry import find_project_root

        base = (
            (Path(override).expanduser().resolve() / "mlperf-tiny-image")
            if override
            else find_project_root() / "data" / "mlperf-tiny-image"
        )
    return {
        "root": base,
        "float_model": base / "pretrainedResnet.tflite",
        "performance_indices": base / "perf_samples_idxs.npy",
    }


def sst2_paths(root: Path | None = None) -> dict[str, Path]:
    if root is not None:
        base = root.resolve()
    else:
        override = os.environ.get("MLPERF_EDU_DATA_DIR")
        from .registry import find_project_root

        base = (
            (Path(override).expanduser().resolve() / "sst2")
            if override
            else find_project_root() / "data" / "sst2"
        )
    return {
        "root": base,
        "dataset": base / "SST-2",
        "zip": base / "SST-2.zip",
        "train": base / "SST-2" / "train.tsv",
        "validation": base / "SST-2" / "dev.tsv",
    }


def ogbn_arxiv_paths(root: Path | None = None) -> dict[str, Path]:
    if root is not None:
        base = root.resolve()
    else:
        override = os.environ.get("MLPERF_EDU_DATA_DIR")
        from .registry import find_project_root

        base = (
            (Path(override).expanduser().resolve() / "ogb")
            if override
            else find_project_root() / "data" / "ogb"
        )
    return {
        "root": base,
        "dataset": base / "ogbn_arxiv",
        "zip": base / "ogbn-arxiv-v1.zip",
    }


def ettm1_paths(root: Path | None = None) -> dict[str, Path]:
    if root is not None:
        base = root.resolve()
    else:
        override = os.environ.get("MLPERF_EDU_DATA_DIR")
        from .registry import find_project_root

        base = (
            (Path(override).expanduser().resolve() / "ettm1")
            if override
            else find_project_root() / "data" / "ettm1"
        )
    return {"root": base, "csv": base / "ETTm1.csv"}


def nanobeir_reranking_paths(root: Path | None = None) -> dict[str, Path]:
    if root is not None:
        base = root.resolve()
    else:
        override = os.environ.get("MLPERF_EDU_DATA_DIR")
        from .registry import find_project_root

        base = (
            (Path(override).expanduser().resolve() / "nanobeir-reranking")
            if override
            else find_project_root() / "data" / "nanobeir-reranking"
        )
    return {"root": base}


def ensure_tinyshakespeare(
    *, download: bool = True, root: Path | None = None
) -> DatasetAsset:
    paths = tinyshakespeare_paths(root)
    base = paths["root"]
    full = paths["full"]
    train = paths["train"]
    val = paths["val"]
    recipe = paths["recipe"]
    base.mkdir(parents=True, exist_ok=True)
    recipe_text = (
        f"source_url={TINY_SHAKESPEARE_URL}\n"
        f"version={TINY_SHAKESPEARE_VERSION}\n"
        f"source_sha256={KARPATHY_TINY_SHAKESPEARE_SHA256}\n"
        "split=train[:90%],validation[90%:]\n"
        "tokenizer=sorted unique characters (65 symbols)\n"
    )

    source_matches = (
        full.exists() and sha256_file(full) == KARPATHY_TINY_SHAKESPEARE_SHA256
    )
    if not source_matches:
        if not download:
            raise FileNotFoundError(
                f"TinyShakespeare is missing at {full}. "
                "Run `mlperf fetch --workload causal-language-modeling --profile max`."
            )
        tmp = full.with_suffix(".download")
        _download(TINY_SHAKESPEARE_URL, tmp)
        if sha256_file(tmp) != KARPATHY_TINY_SHAKESPEARE_SHA256:
            tmp.unlink(missing_ok=True)
            raise ValueError("Tiny Shakespeare source SHA-256 does not match nanoGPT")
        tmp.replace(full)

    recipe_mismatch = (
        not recipe.exists() or recipe.read_text(encoding="utf-8") != recipe_text
    )
    if recipe_mismatch or not train.exists() or not val.exists():
        text = full.read_text(encoding="utf-8")
        split_idx = int(len(text) * 0.9)
        train.write_text(text[:split_idx], encoding="utf-8")
        val.write_text(text[split_idx:], encoding="utf-8")
        recipe.write_text(recipe_text, encoding="utf-8")

    files = [full, train, val, recipe]
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
        normalized = (
            normalized[line_end + 1 :]
            if line_end != -1
            else normalized[start + len(start_marker) :]
        )
    end = normalized.find(end_marker)
    if end != -1:
        normalized = normalized[:end]
    normalized = normalized.strip() + "\n"
    return normalized[:TINY_SHAKESPEARE_TARGET_CHARS]


def ensure_cifar10(*, download: bool = True, root: Path | None = None) -> DatasetAsset:
    paths = cifar10_paths(root)
    base = paths["root"]
    base.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    for relative_name, expected_sha256 in CIFAR10_HF_FILES.items():
        destination = base / relative_name
        if not destination.is_file() or sha256_file(destination) != expected_sha256:
            if not download:
                raise FileNotFoundError(
                    f"CIFAR-10 is missing at {destination}. "
                    "Run `mlperf fetch --workload image-classification --profile max`."
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            url = (
                f"https://huggingface.co/datasets/{CIFAR10_HF_REPO_ID}/resolve/"
                f"{CIFAR10_HF_REVISION}/{relative_name}"
            )
            tmp = destination.with_suffix(".download")
            _download(url, tmp)
            if sha256_file(tmp) != expected_sha256:
                tmp.unlink(missing_ok=True)
                raise ValueError(f"CIFAR-10 Parquet SHA-256 mismatch: {relative_name}")
            tmp.replace(destination)
        files.append(destination)
    load_cifar10_dataset(root=base, train=False, download=False)
    files_tuple = tuple(files)
    n_bytes = sum(path.stat().st_size for path in files)
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(base)).encode("utf-8") + b"\0")
        digest.update(sha256_file(path).encode("ascii") + b"\n")

    return DatasetAsset(
        name="cifar10",
        root=base,
        files=files_tuple,
        sha256=f"sha256:{digest.hexdigest()}",
        n_bytes=n_bytes,
        source=f"https://huggingface.co/datasets/{CIFAR10_HF_REPO_ID}/tree/{CIFAR10_HF_REVISION}",
    )


def ensure_fashion_mnist(
    *, download: bool = True, root: Path | None = None
) -> DatasetAsset:
    paths = fashion_mnist_paths(root)
    base = paths["root"]
    base.mkdir(parents=True, exist_ok=True)

    try:
        load_fashion_mnist_dataset(root=base, train=True, download=download)
        load_fashion_mnist_dataset(root=base, train=False, download=download)
    except Exception as exc:
        if not download:
            raise FileNotFoundError(
                f"Fashion-MNIST is missing at {base}. Run the standalone "
                "optimization lab once with network access to prepare it."
            ) from exc
        raise

    files = tuple(
        sorted(path for path in (base / "FashionMNIST").rglob("*") if path.is_file())
    )
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


def ensure_mlperf_tiny_kws(
    *, download: bool = True, root: Path | None = None
) -> DatasetAsset:
    paths = mlperf_tiny_kws_paths(root)
    base = paths["root"]
    dataset = paths["dataset"]
    base.mkdir(parents=True, exist_ok=True)

    archive = paths["archive"]
    if not (dataset / "y_labels.csv").is_file():
        if not download:
            raise FileNotFoundError(
                f"MLPerf Tiny KWS evaluation data is missing at {dataset}. "
                "Run `mlperf fetch --workload keyword-spotting --profile max`."
            )
        _download(EEMBC_RUNNER_ARCHIVE_URL, archive)
        if sha256_file(archive) != EEMBC_RUNNER_ARCHIVE_SHA256:
            archive.unlink(missing_ok=True)
            raise ValueError(
                "EEMBC runner archive SHA-256 does not match the pinned value"
            )
        prefix = f"energyrunner-{EEMBC_RUNNER_COMMIT}/datasets/kws01/"
        with tarfile.open(archive, "r:gz") as tf:
            members = [
                member for member in tf.getmembers() if member.name.startswith(prefix)
            ]
            if not members:
                raise FileNotFoundError(
                    "Pinned EEMBC archive has no datasets/kws01 files"
                )
            for member in members:
                if not member.isfile():
                    continue
                relative = Path(member.name.removeprefix(prefix))
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"unsafe EEMBC archive member: {member.name}")
                target = dataset / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tf.extractfile(member)
                if source is None:
                    raise FileNotFoundError(
                        f"could not read EEMBC archive member: {member.name}"
                    )
                with source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)

    model_specs = (
        (
            paths["float_model"],
            f"{MLPERF_TINY_KWS_MODEL_BASE_URL}/kws_ref_model_float32.tflite",
            MLPERF_TINY_KWS_FLOAT_MODEL_SHA256,
        ),
        (
            paths["int8_model"],
            f"{MLPERF_TINY_KWS_MODEL_BASE_URL}/kws_ref_model.tflite",
            MLPERF_TINY_KWS_INT8_MODEL_SHA256,
        ),
    )
    for model_path, url, expected_sha256 in model_specs:
        if not model_path.is_file() or sha256_file(model_path) != expected_sha256:
            if not download:
                raise FileNotFoundError(
                    f"Pinned MLPerf Tiny KWS model is missing at {model_path}"
                )
            _download(url, model_path)
        if sha256_file(model_path) != expected_sha256:
            model_path.unlink(missing_ok=True)
            raise ValueError(
                f"MLPerf Tiny KWS model SHA-256 mismatch: {model_path.name}"
            )

    files = tuple(sorted(path for path in dataset.rglob("*") if path.is_file()))
    if len(files) != 1001:
        raise ValueError(
            f"MLPerf Tiny KWS evaluation set expected 1001 files, found {len(files)}"
        )
    n_bytes = sum(path.stat().st_size for path in files)
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(dataset)).encode("utf-8") + b"\0")
        digest.update(sha256_file(path).encode("ascii") + b"\n")

    return DatasetAsset(
        name="mlperf-tiny-kws-eval",
        root=dataset,
        files=files,
        sha256=f"sha256:{digest.hexdigest()}",
        n_bytes=n_bytes,
        source=EEMBC_RUNNER_ARCHIVE_URL,
    )


def ensure_mlperf_tiny_vww(
    *, download: bool = True, root: Path | None = None
) -> DatasetAsset:
    """Prepare the exact 1,000-image MLPerf Tiny VWW accuracy set."""
    paths = mlperf_tiny_vww_paths(root)
    base = paths["root"]
    dataset = paths["dataset"]
    base.mkdir(parents=True, exist_ok=True)

    source_archive = paths["source_archive"]
    if (
        not source_archive.is_file()
        or source_archive.stat().st_size != MLPERF_TINY_VWW_ARCHIVE_BYTES
        or sha256_file(source_archive) != MLPERF_TINY_VWW_ARCHIVE_SHA256
    ):
        if not download:
            raise FileNotFoundError(
                f"MLPerf Tiny VWW source archive is missing at {source_archive}. "
                "Run `mlperf fetch --workload visual-wake-words --profile max`."
            )
        source_archive.unlink(missing_ok=True)
        _download(MLPERF_TINY_VWW_ARCHIVE_URL, source_archive)
    if (
        source_archive.stat().st_size != MLPERF_TINY_VWW_ARCHIVE_BYTES
        or sha256_file(source_archive) != MLPERF_TINY_VWW_ARCHIVE_SHA256
    ):
        source_archive.unlink(missing_ok=True)
        raise ValueError("MLPerf Tiny VWW source archive does not match its pin")

    runner_archive = paths["runner_archive"]
    if (
        not runner_archive.is_file()
        or sha256_file(runner_archive) != EEMBC_RUNNER_ARCHIVE_SHA256
    ):
        if not download:
            raise FileNotFoundError(
                f"Pinned EEMBC VWW labels are missing at {runner_archive}. "
                "Run `mlperf fetch --workload visual-wake-words --profile max`."
            )
        runner_archive.unlink(missing_ok=True)
        _download(EEMBC_RUNNER_ARCHIVE_URL, runner_archive)
    if sha256_file(runner_archive) != EEMBC_RUNNER_ARCHIVE_SHA256:
        runner_archive.unlink(missing_ok=True)
        raise ValueError("EEMBC runner archive does not match its pinned SHA-256")

    current_files = tuple(sorted(path for path in dataset.rglob("*") if path.is_file()))
    current_digest = _dataset_file_digest(dataset, current_files)
    if (
        len(current_files) != 1001
        or sum(path.stat().st_size for path in current_files)
        != MLPERF_TINY_VWW_DATASET_BYTES
        or current_digest != MLPERF_TINY_VWW_DATASET_SHA256
    ):
        staging = base / "vww01.staging"
        shutil.rmtree(staging, ignore_errors=True)
        images = staging / "images"
        images.mkdir(parents=True)

        label_member = f"energyrunner-{EEMBC_RUNNER_COMMIT}/datasets/vww01/y_labels.csv"
        with tarfile.open(runner_archive, "r:gz") as tf:
            try:
                member = tf.getmember(label_member)
            except KeyError as exc:
                raise FileNotFoundError(
                    "Pinned EEMBC archive has no VWW label index"
                ) from exc
            source = tf.extractfile(member)
            if source is None:
                raise FileNotFoundError("Could not read the EEMBC VWW label index")
            labels_bytes = source.read()
        if hashlib.sha256(labels_bytes).hexdigest() != MLPERF_TINY_VWW_LABELS_SHA256:
            raise ValueError("EEMBC VWW label index does not match its pin")
        labels_path = staging / "y_labels.csv"
        labels_path.write_bytes(labels_bytes)

        labels: dict[str, int] = {}
        with labels_path.open(newline="") as handle:
            for row in csv.reader(handle):
                if len(row) != 3:
                    raise ValueError(f"invalid MLPerf Tiny VWW label row: {row}")
                stem = Path(row[0].strip()).stem
                label = int(row[2])
                if stem in labels or label not in {0, 1}:
                    raise ValueError(f"invalid MLPerf Tiny VWW label entry: {row}")
                labels[stem] = label
        if len(labels) != 1000 or sum(labels.values()) != 500:
            raise ValueError(
                "MLPerf Tiny VWW labels must contain 1,000 balanced examples"
            )

        found: dict[str, str] = {}
        with tarfile.open(source_archive, "r|gz") as tf:
            for member in tf:
                if not member.isfile() or not member.name.endswith(".jpg"):
                    continue
                archive_path = Path(member.name)
                stem = archive_path.stem.rsplit("_", 1)[-1]
                if stem not in labels:
                    continue
                if stem in found:
                    raise ValueError(
                        f"multiple VWW archive images resolve to label {stem}"
                    )
                expected_class = "person" if labels[stem] == 1 else "non_person"
                if (
                    len(archive_path.parts) < 3
                    or archive_path.parts[-2] != expected_class
                ):
                    raise ValueError(
                        f"MLPerf Tiny VWW class mismatch for {member.name}"
                    )
                source = tf.extractfile(member)
                if source is None:
                    raise FileNotFoundError(
                        f"could not read VWW archive member: {member.name}"
                    )
                with source, (images / f"{stem}.jpg").open("wb") as destination:
                    shutil.copyfileobj(source, destination)
                found[stem] = member.name
        missing = sorted(set(labels) - set(found))
        if missing:
            raise FileNotFoundError(
                f"MLPerf Tiny VWW archive is missing {len(missing)} indexed images"
            )

        staged_files = tuple(
            sorted(path for path in staging.rglob("*") if path.is_file())
        )
        staged_digest = _dataset_file_digest(staging, staged_files)
        staged_bytes = sum(path.stat().st_size for path in staged_files)
        if (
            len(staged_files) != 1001
            or staged_bytes != MLPERF_TINY_VWW_DATASET_BYTES
            or staged_digest != MLPERF_TINY_VWW_DATASET_SHA256
        ):
            raise ValueError("prepared MLPerf Tiny VWW dataset does not match its pin")
        shutil.rmtree(dataset, ignore_errors=True)
        staging.replace(dataset)

    model_specs = (
        (
            paths["float_model"],
            f"{MLPERF_TINY_VWW_MODEL_BASE_URL}/vww_96_float.tflite",
            MLPERF_TINY_VWW_FLOAT_MODEL_SHA256,
        ),
        (
            paths["int8_model"],
            f"{MLPERF_TINY_VWW_MODEL_BASE_URL}/vww_96_int8.tflite",
            MLPERF_TINY_VWW_INT8_MODEL_SHA256,
        ),
    )
    for model_path, url, expected_sha256 in model_specs:
        if not model_path.is_file() or sha256_file(model_path) != expected_sha256:
            if not download:
                raise FileNotFoundError(
                    f"Pinned MLPerf Tiny VWW model is missing at {model_path}"
                )
            model_path.unlink(missing_ok=True)
            _download(url, model_path)
        if sha256_file(model_path) != expected_sha256:
            model_path.unlink(missing_ok=True)
            raise ValueError(
                f"MLPerf Tiny VWW model SHA-256 mismatch: {model_path.name}"
            )

    files = tuple(sorted(path for path in dataset.rglob("*") if path.is_file()))
    return DatasetAsset(
        name="mlperf-tiny-vww-eval",
        root=dataset,
        files=files,
        sha256=f"sha256:{MLPERF_TINY_VWW_DATASET_SHA256}",
        n_bytes=MLPERF_TINY_VWW_DATASET_BYTES,
        source=MLPERF_TINY_VWW_ARCHIVE_URL,
    )


def _dataset_file_digest(root: Path, files: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode("utf-8") + b"\0")
        digest.update(sha256_file(path).encode("ascii") + b"\n")
    return digest.hexdigest()


def ensure_mlperf_tiny_image(
    *, download: bool = True, root: Path | None = None
) -> DatasetAsset:
    paths = mlperf_tiny_image_paths(root)
    base = paths["root"]
    base.mkdir(parents=True, exist_ok=True)
    specifications = (
        (
            paths["float_model"],
            f"{MLPERF_TINY_IMAGE_BASE_URL}/trained_models/pretrainedResnet.tflite",
            MLPERF_TINY_IMAGE_FLOAT_MODEL_SHA256,
        ),
        (
            paths["performance_indices"],
            f"{MLPERF_TINY_IMAGE_BASE_URL}/perf_samples_idxs.npy",
            MLPERF_TINY_IMAGE_PERF_INDICES_SHA256,
        ),
    )
    for destination, url, expected_sha256 in specifications:
        if destination.is_file() and sha256_file(destination) == expected_sha256:
            continue
        if not download:
            raise FileNotFoundError(
                f"MLPerf Tiny image evaluation asset is missing at {destination}. "
                "Run `mlperf fetch --workload image-classification --profile max`."
            )
        tmp = destination.with_suffix(destination.suffix + ".download")
        _download(url, tmp)
        if sha256_file(tmp) != expected_sha256:
            tmp.unlink(missing_ok=True)
            raise ValueError(
                f"MLPerf Tiny image asset failed SHA-256 verification: {destination.name}"
            )
        tmp.replace(destination)

    files = tuple(specification[0] for specification in specifications)
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8") + b"\0")
        digest.update(sha256_file(path).encode("ascii") + b"\n")
    return DatasetAsset(
        name="mlperf-tiny-image-evaluation",
        root=base,
        files=files,
        sha256=f"sha256:{digest.hexdigest()}",
        n_bytes=sum(path.stat().st_size for path in files),
        source=MLPERF_TINY_IMAGE_BASE_URL,
    )


def ensure_sst2(*, download: bool = True, root: Path | None = None) -> DatasetAsset:
    paths = sst2_paths(root)
    base = paths["root"]
    dataset = paths["dataset"]
    archive = paths["zip"]
    base.mkdir(parents=True, exist_ok=True)
    if not paths["validation"].is_file():
        if not download:
            raise FileNotFoundError(
                f"GLUE SST-2 is missing at {dataset}. "
                "Run `mlperf fetch --workload text-classification --profile max`."
            )
        _download(GLUE_SST2_URL, archive)
        if sha256_file(archive) != GLUE_SST2_ZIP_SHA256:
            archive.unlink(missing_ok=True)
            raise ValueError(
                "GLUE SST-2 archive SHA-256 does not match the pinned value"
            )
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                relative = Path(info.filename)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(
                        f"unsafe GLUE SST-2 archive member: {info.filename}"
                    )
            zf.extractall(base)

    files = (paths["validation"],)
    if not paths["validation"].is_file():
        raise FileNotFoundError(f"GLUE SST-2 extraction is incomplete under {dataset}")
    n_bytes = sum(path.stat().st_size for path in files)
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(dataset)).encode("utf-8") + b"\0")
        digest.update(sha256_file(path).encode("ascii") + b"\n")
    return DatasetAsset(
        name="sst2",
        root=dataset,
        files=files,
        sha256=f"sha256:{digest.hexdigest()}",
        n_bytes=n_bytes,
        source=GLUE_SST2_URL,
    )


def ensure_ogbn_arxiv(
    *, download: bool = True, root: Path | None = None
) -> DatasetAsset:
    paths = ogbn_arxiv_paths(root)
    base = paths["root"]
    dataset = paths["dataset"]
    archive = paths["zip"]
    release_marker = dataset / "RELEASE_v1.txt"
    base.mkdir(parents=True, exist_ok=True)
    if not release_marker.is_file():
        if not download:
            raise FileNotFoundError(
                f"ogbn-arxiv is missing at {dataset}. "
                "Run `mlperf fetch --workload graph-node-classification --profile max`."
            )
        _download(OGBN_ARXIV_URL, archive)
        if sha256_file(archive) != OGBN_ARXIV_ZIP_SHA256:
            archive.unlink(missing_ok=True)
            raise ValueError(
                "ogbn-arxiv archive SHA-256 does not match the pinned value"
            )
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                relative = Path(info.filename)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(
                        f"unsafe ogbn-arxiv archive member: {info.filename}"
                    )
            zf.extractall(base)
        extracted = base / "arxiv"
        if dataset.exists():
            shutil.rmtree(dataset)
        extracted.replace(dataset)

    if not archive.is_file() or sha256_file(archive) != OGBN_ARXIV_ZIP_SHA256:
        if not download:
            raise FileNotFoundError(
                f"pinned ogbn-arxiv archive is missing at {archive}"
            )
        _download(OGBN_ARXIV_URL, archive)
        if sha256_file(archive) != OGBN_ARXIV_ZIP_SHA256:
            archive.unlink(missing_ok=True)
            raise ValueError(
                "ogbn-arxiv archive SHA-256 does not match the pinned value"
            )

    return DatasetAsset(
        name="ogbn-arxiv",
        root=base,
        files=(archive,),
        sha256=f"sha256:{OGBN_ARXIV_ZIP_SHA256}",
        n_bytes=archive.stat().st_size,
        source=OGBN_ARXIV_URL,
    )


def ensure_ettm1(*, download: bool = True, root: Path | None = None) -> DatasetAsset:
    paths = ettm1_paths(root)
    base = paths["root"]
    csv_path = paths["csv"]
    base.mkdir(parents=True, exist_ok=True)
    if not csv_path.is_file() or sha256_file(csv_path) != ETTM1_SHA256:
        if not download:
            raise FileNotFoundError(
                f"ETTm1 is missing at {csv_path}. "
                "Run `mlperf fetch --workload time-series-forecasting --profile max`."
            )
        tmp = csv_path.with_suffix(".download")
        _download(ETTM1_URL, tmp)
        if sha256_file(tmp) != ETTM1_SHA256:
            tmp.unlink(missing_ok=True)
            raise ValueError("ETTm1 CSV SHA-256 does not match the pinned value")
        tmp.replace(csv_path)
    return DatasetAsset(
        name="ettm1",
        root=base,
        files=(csv_path,),
        sha256=f"sha256:{ETTM1_SHA256}",
        n_bytes=csv_path.stat().st_size,
        source=ETTM1_URL,
    )


def ensure_nanobeir_reranking(
    *, download: bool = True, root: Path | None = None
) -> DatasetAsset:
    base = nanobeir_reranking_paths(root)["root"]
    base.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    for relative_name, expected_sha256 in NANOBEIR_RERANKING_FILES.items():
        destination = base / relative_name
        if not destination.is_file() or sha256_file(destination) != expected_sha256:
            if not download:
                raise FileNotFoundError(
                    f"NanoBEIR reranking asset is missing at {destination}. "
                    "Run `mlperf fetch --workload information-retrieval --profile max`."
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            url = (
                f"https://huggingface.co/datasets/{NANOBEIR_REPO_ID}/resolve/"
                f"{NANOBEIR_REVISION}/{relative_name}"
            )
            tmp = destination.with_suffix(".download")
            _download(url, tmp)
            if sha256_file(tmp) != expected_sha256:
                tmp.unlink(missing_ok=True)
                raise ValueError(f"NanoBEIR SHA-256 mismatch: {relative_name}")
            tmp.replace(destination)
        files.append(destination)
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(base)).encode("utf-8") + b"\0")
        digest.update(sha256_file(path).encode("ascii") + b"\n")
    return DatasetAsset(
        name="nanobeir-reranking",
        root=base,
        files=tuple(files),
        sha256=f"sha256:{digest.hexdigest()}",
        n_bytes=sum(path.stat().st_size for path in files),
        source=f"https://huggingface.co/datasets/{NANOBEIR_REPO_ID}/tree/{NANOBEIR_REVISION}",
    )


def _download(url: str, destination: Path) -> None:
    try:
        urllib.request.urlretrieve(url, destination)
        return
    except Exception as urllib_exc:
        if shutil.which("curl"):
            try:
                subprocess.run(
                    [
                        "curl",
                        "--fail",
                        "--location",
                        "--silent",
                        "--show-error",
                        url,
                        "--output",
                        str(destination),
                    ],
                    check=True,
                )
                return
            except subprocess.CalledProcessError as curl_exc:
                if destination.exists():
                    destination.unlink()
                raise RuntimeError(
                    f"failed to download {url} with urllib and curl"
                ) from curl_exc
        if destination.exists():
            destination.unlink()
        raise RuntimeError(f"failed to download {url}") from urllib_exc
