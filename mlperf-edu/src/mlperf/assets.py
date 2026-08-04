from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import uuid
import zipfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from platformdirs import user_cache_path


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
MLPERF_TINY_IMAGE_EVALUATOR_SHA256 = (
    "64be48b7cfe4ca1157f01227f3122ee1760c8c5509c6875cacaab8ff931b22d1"
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
MLPERF_TINY_ANOMALY_COMMIT = "4addd0fa08d216e20637637874e084895f289da4"
MLPERF_TINY_ANOMALY_BASE_URL = (
    f"https://raw.githubusercontent.com/mlcommons/tiny/{MLPERF_TINY_ANOMALY_COMMIT}"
)
MLPERF_TINY_ANOMALY_MODEL_BASE_URL = f"{MLPERF_TINY_ANOMALY_BASE_URL}/benchmark/training/anomaly_detection/trained_models"
MLPERF_TINY_ANOMALY_LABELS_URL = (
    f"{MLPERF_TINY_ANOMALY_BASE_URL}/benchmark/evaluation/datasets/ad01/y_labels.csv"
)
MLPERF_TINY_ANOMALY_ARCHIVE_URL = (
    "https://zenodo.org/api/records/3678171/files/dev_data_ToyCar.zip/content"
)
MLPERF_TINY_ANOMALY_ARCHIVE_MD5 = "4dec75ca8d9f666aa9e4c1894a740501"
MLPERF_TINY_ANOMALY_ARCHIVE_BYTES = 1_816_443_231
MLPERF_TINY_ANOMALY_LABELS_SHA256 = (
    "4ecd91868e197ee0a6739d4bd7abde73eac2fa31e9a88d6bc6aedefa136ff2a4"
)
MLPERF_TINY_ANOMALY_MEMBER_MANIFEST_SHA256 = (
    "91e7f55220f5fb7c0d6d6d855e47b79fbd72fcc29766848a9750a73cbfe8a9d9"
)
MLPERF_TINY_ANOMALY_DATASET_SHA256 = (
    "a7953bc0ad0caffd642dbdba4d0ea467515ba5480f7e0c46b62dfbe8ae61f716"
)
MLPERF_TINY_ANOMALY_DATASET_BYTES = 25_408_204
MLPERF_TINY_ANOMALY_FLOAT_MODEL_SHA256 = (
    "c66636f4d7f8af8b10518e7be750a22c9d8d46ec97326b40b0d94c097e0aad9b"
)
MLPERF_TINY_ANOMALY_INT8_MODEL_SHA256 = (
    "87cf24194ef93d1d9b11a591d805526b98008e351655d29883c825c9c106ba24"
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
HUMANEVAL_PLUS_VERSION = "v0.1.10"
HUMANEVAL_PLUS_URL = (
    "https://github.com/evalplus/humanevalplus_release/releases/download/"
    f"{HUMANEVAL_PLUS_VERSION}/HumanEvalPlus.jsonl.gz"
)
HUMANEVAL_PLUS_ARCHIVE_SHA256 = (
    "272720b90ac375502c8ed23cd791c2a93dfb22a911641a494da74a426c09f101"
)
HUMANEVAL_PLUS_SHA256 = (
    "42526ec0e7d5f3ee0b06d6ced98f8c8bae3d76519151bfb3d36f79010645bd7f"
)
HUMANEVAL_PLUS_BYTES = 7_714_666
EVALPLUS_COMMIT = "899b2b31bbe8d6e12337b865a8aa03fcd57c1121"
EVALPLUS_ARCHIVE_URL = (
    f"https://github.com/evalplus/evalplus/archive/{EVALPLUS_COMMIT}.tar.gz"
)
EVALPLUS_ARCHIVE_SHA256 = (
    "d3a5ce49566224a054debc2b51f9290e070734841044b0bb6764f92c376e8149"
)
BFCL_COMMIT = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
BFCL_ARCHIVE_URL = (
    f"https://github.com/ShishirPatil/gorilla/archive/{BFCL_COMMIT}.tar.gz"
)
BFCL_ARCHIVE_SHA256 = "c6b5081337cabd317b56c6eead2ec735f7a3cb86ac4fd664e3f9c8c02d1f1f1e"
BFCL_EVALUATOR_COMMIT = "f7cf7359b7ac615a0b294831c5ba2bc95ee4a000"
BFCL_EVALUATOR_FILES = {
    "bfcl_eval/constants/enums.py": "2182becfa2a1d071ee1db30db593b4758c6bf866aa12d2d4b8daf09175ea518a",
    "bfcl_eval/constants/type_mappings.py": "1702fb67afbe2c492608e58e2b7d02e46381f50166b47f3c952f76e34c7cd3bd",
    "bfcl_eval/eval_checker/ast_eval/ast_checker.py": "2aae7a68461a8f76c0be3894c8901b66b56967a1989d3ab066051e3fb97f1538",
    "bfcl_eval/eval_checker/ast_eval/type_convertor/java_type_converter.py": "2fd4f4b0443b3dd974a1723bb4e45c086d7b352631062da7807ad1ad40706604",
    "bfcl_eval/eval_checker/ast_eval/type_convertor/js_type_converter.py": "a114e9ff75c025cb52787ac33d6c2fbaa390905c6125a2b3c6afebab232bb5e4",
    "bfcl_eval/model_handler/local_inference/qwen_fc.py": "e6d0ac52783a595b6627323eef486e18790a7b1084658063a79dc45f0ddbf474",
}
BFCL_DATA_FILES = {
    "BFCL_v4_multiple.json": "aef168155ebd74b7ac2401198b201343bc7d16d7a3d7e0d4e6d8ee82c6969b2a",
    "BFCL_v4_parallel.json": "19f51a82eff42e5d62541aa500115a056eb78f437c2ba1f10415fd7c8e5dda84",
    "BFCL_v4_parallel_multiple.json": "8863ea8433239f55c5f016154cf0830853c89f693c6ea270396a2fa121960579",
    "BFCL_v4_simple_java.json": "13d2303a125b08754f0e41995b9273b5005fa8ed8ebfaa24ef53b4d83c4b5c6e",
    "BFCL_v4_simple_javascript.json": "329e67fedf79a6243d93dbda4b388d12bd2d31f1f2163d92cb6ef676d1764f44",
    "BFCL_v4_simple_python.json": "82dd63ba502eb2520c6b5d1d9a5c4b590e03ff261565175561f6228a367d1991",
    "possible_answer/BFCL_v4_multiple.json": "244e00ce9395df948bcafc7bee64e8f9c87ef70887587d83cae45b13699f3047",
    "possible_answer/BFCL_v4_parallel.json": "8a6aa19c1adddc6a5a2f7e40f9dbf30cc7e95815e7b830c90589ab318229e0f0",
    "possible_answer/BFCL_v4_parallel_multiple.json": "5ebf24f458c1f16300c05505d83d6f0a1b68b79be273a033febd0d4f840507e3",
    "possible_answer/BFCL_v4_simple_java.json": "78f25616084044fa05bbfcee68e03f6ececb222bdd5cb3b7783a675fb3366e35",
    "possible_answer/BFCL_v4_simple_javascript.json": "e2f9f2e51d88e0c8056ffbf1a3dd3d02eb032532d2b5d98c9cc9003385bdd56b",
    "possible_answer/BFCL_v4_simple_python.json": "90cd5bc653690ee8e459b5b3f3fc9458606f7f3fcbf795bb51b7dc581f8c86dc",
}
EDM_COMMIT = "008a4e5316c8e3bfe61a62f874bddba254295afb"
EDM_CIFAR10_CHECKPOINT_URL = (
    "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl"
)
EDM_CIFAR10_CHECKPOINT_SHA256 = (
    "b27385e4f2f8d4d1e2e2c03864c5a9574a0c58ba81eb11625ae6d15c401f4796"
)
EDM_CIFAR10_CHECKPOINT_BYTES = 223_183_453
EDM_CIFAR10_FID_REFERENCE_URL = (
    "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz"
)
EDM_CIFAR10_FID_REFERENCE_SHA256 = (
    "4ada108cdfd8e43409ed427f5f76631beac9b9617cdcaf416d41b1e161d85969"
)
EDM_ARCHIVE_URL = f"https://github.com/NVlabs/edm/archive/{EDM_COMMIT}.tar.gz"
EDM_ARCHIVE_SHA256 = "9ddd0bddaf75a9e066c526f13a882958b0c0b7a1332df2d32f6a1c46646fa746"
EDM_INCEPTION_URL = (
    "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/"
    "files/metrics/inception-2015-12-05.pkl"
)
EDM_INCEPTION_SHA256 = (
    "a31bd1d4522101109044ec28621a5fcb591ed6115e0d58104fc013fa01ef94f7"
)
EDM_INCEPTION_BYTES = 95_617_399
EDM_SOURCE_FILES = {
    "dnnlib/util.py": "00a0e339dc8eca358c9053ae6d93ab87b8d3f4df1e5dd30d3185558721986e1c",
    "fid.py": "704be0de2c77c0090fdb9787268360ef9d438ca83c6aca14aa6d4f51dcff364d",
    "generate.py": "1af4e135d3a17f7cba1e022f22260d4e7534ea525e6162783e00805c53204c2a",
    "torch_utils/misc.py": "d1b9ccfa41cbb0e232675d49d1875179cb010dbbd2a021584aa5d1614348ac83",
    "torch_utils/persistence.py": "71589e7a4b5175047a48a4bc58e30438b58bea87e415e6873d13d7d438c44b38",
    "training/dataset.py": "f46fe15ecd66206819e416ffca6ea22a06610405d6954ccc0fffc53092ab81c2",
    "training/networks.py": "5db27dcd96674b95c72d5e6491b879cdc35e24039ada3411b4b46a28ed1fe284",
}
DLRM_INFERENCE_COMMIT = "8b58587c93af2a5ee67722064f2540a2db15d42f"
DLRM_INFERENCE_ARCHIVE_URL = (
    f"https://github.com/mlcommons/inference/archive/{DLRM_INFERENCE_COMMIT}.tar.gz"
)
DLRM_INFERENCE_ARCHIVE_SHA256 = (
    "9a0f82844cc48a05face9384d200d260da1120aa86f0f4bb140677daaf84b6d2"
)
DLRM_IMPLEMENTATION_COMMIT = "6d75c84d834380a365e2f03d4838bee464157516"
DLRM_IMPLEMENTATION_ARCHIVE_URL = f"https://github.com/facebookresearch/dlrm/archive/{DLRM_IMPLEMENTATION_COMMIT}.tar.gz"
DLRM_IMPLEMENTATION_ARCHIVE_SHA256 = (
    "748b2f5e62231dbe5965f30b85caaefcb66e8dffeda394f68e74770beb73b5af"
)
DLRM_INFERENCE_FILES = {
    "mlperf.conf": "4e84d49c30915386e7d5c9b02ee06c166d997dbf3883ee479c17dbeefe68a81c",
    "recommendation/dlrm/pytorch/python/backend_pytorch_native.py": "c88931a6f050208d50b116ca8bc16b85818f5d7bc8b5f0c8ee3285da65e625f7",
    "recommendation/dlrm/pytorch/python/criteo.py": "ff304bb29baead0f0990050785b2aa92c45c5a4e2b63bb52bcdcac6e746c1883",
    "recommendation/dlrm/pytorch/python/main.py": "258a2cb45aa61ee758f37abd6affb6b0ede3709460a7751d2cce861680eae82d",
    "recommendation/dlrm/pytorch/tools/accuracy-dlrm.py": "304fa14c4b47aceaa651a135d01fa71798640ee09aa1ce01e468558021403919",
    "recommendation/dlrm/pytorch/tools/dist_quantile.txt": "76454af11405bcfa9e375c79a17738b6fcb7633a8467da2ce785eb3fab8ec80b",
    "recommendation/dlrm/pytorch/user.conf": "8e3771762aaeccf2e29cdae92a7af2f27ffe0923dfe32675080ead72585b01c9",
}
DLRM_IMPLEMENTATION_FILES = {
    "data_loader_terabyte.py": "867ef30aaa68b67c29f01f5da5ef81b9d5a0b4aa39795d35678e47d3d75049b1",
    "data_utils.py": "a4285cc250152491fcad182f1c01806a0b0260cd0f2f57b86e4f44743f5c3aea",
    "dlrm_data_pytorch.py": "5dd70d7322aa91fdf8b00fe3dcb3f8a380c28cd499498828fa9cff5bfd72d61e",
    "dlrm_s_pytorch.py": "8d78c9ca22ef365d3884e868b2efb3ebef9e7bb9042d3c661fb55b7117c58c11",
}
DLRM_CHECKPOINT_URL = "https://dlrm.s3-us-west-1.amazonaws.com/models/tb00_40M.pt"
DLRM_CHECKPOINT_MD5 = "2d49a5288cddb37c3c64860a06d79bb9"
DLRM_TRAINING_SUBMODULE_COMMIT = "8e7ad54541aeda54a8e5152732b9fb293a22b10c"
MINIGO_COMMIT = "0badcd1786fcb007725ed05f1c44e9d80bbeac52"
MINIGO_ARCHIVE_URL = (
    f"https://github.com/mlcommons/training/archive/{MINIGO_COMMIT}.tar.gz"
)
MINIGO_ARCHIVE_SHA256 = (
    "d91b694f06adcfb67085c9d9aff44a8c6df728c6bdbae75bfb187dd010adcfd9"
)
MINIGO_ARCHIVE_BYTES = 4_679_016
MINIGO_SOURCE_FILES = {
    "reinforcement/README.md": "ab25b2b06f012b5804c1750f1077411fb95e8356dc23de91ff0ab49dab982bf4",
    "reinforcement/tensorflow/Dockerfile": "609419190d6c513b674da8520c6eb637ad125e56d01daaf3f893b84cba389247",
    "reinforcement/tensorflow/minigo/benchmark_sgf/9x9_pro_IYHN.sgf": "486a6c67885354dfa31bcc16fc4a17e83ada40b9fd43600b4245b04e3df09314",
    "reinforcement/tensorflow/minigo/benchmark_sgf/9x9_pro_IYMD.sgf": "654fd67a1cfc0ed7278eed8e766ba8b90b232f418173523c26dc7efffdb7adc9",
    "reinforcement/tensorflow/minigo/benchmark_sgf/9x9_pro_YKSH.sgf": "5cb52058e21284b7b0306f29462b9ca09d4da729c7eebf25b88ce9291fbcb571",
    "reinforcement/tensorflow/minigo/benchmark_sgf/9x9_pro_YSIY.sgf": "80077ea05c80f8f21208a1dc7bfc563753344ece65ea43fbb9e20a2d6cd09965",
    "reinforcement/tensorflow/minigo/dual_net.py": "15b096eca61e96831c4df7e3bd5fd849fc9fe15a8c1d80dee68c4565be747ddb",
    "reinforcement/tensorflow/minigo/loop_init.py": "7b204a7a73b2472c2ff7691abdf7c7a7fb8df029bc1dafdacaf426b5b0a8caa1",
    "reinforcement/tensorflow/minigo/loop_main.sh": "a57ca8f8dc915817dbcb975875bd76f63d35c64d4f4e47776839dedb36d022bd",
    "reinforcement/tensorflow/minigo/loop_selfplay.py": "7e6a417fc9da4f132500535e4a3c8315eda43603859eaaa2dec8a38425415782",
    "reinforcement/tensorflow/minigo/loop_train_eval.py": "b97a30269fc120a5e410bf2126add99a8b0f80d9f356b8900bfdb4962c266b0a",
    "reinforcement/tensorflow/minigo/params/final.json": "ba76de9fdf3e2bb261537b04e730fa4214ca92e08a2677e64d3bf1994841acc0",
    "reinforcement/tensorflow/minigo/predict_games.py": "e94b146eb66b1de63debf10f601e8c861bf77b0738efbbced70627844cde5e47",
    "reinforcement/tensorflow/run.sh": "57c4573eea407502c3c42d3e224a70cf5332d5bd538b48d5f6d348dda656ea86",
    "reinforcement/tensorflow/run_and_time.sh": "f778c840cc0d7581ced23fbb265cd8f15d804f0d385a167f729d8d62bd51a51f",
}


def dlrm_environment_handoff_contract() -> dict[str, Any]:
    """Return the portable environment contract for DLRM quality execution."""
    return {
        "schema": "mlperf-edu-environment-handoff/0.1",
        "workload": "recommendation",
        "profile": "max",
        "execution_status": "environment-gated-quality-conformance",
        "quality": {
            "metric": "roc_auc",
            "target": 0.8025,
            "direction": "higher",
            "acceptance_runs": 1,
        },
        "required_hardware": {
            "system": "single-node",
            "recommended_host_memory_gib": 256,
            "device_choices": ["cpu", "gpu"],
            "gpu_requirement": "CUDA-visible PyTorch when device is gpu",
        },
        "external_assets": {
            "dataset": {
                "name": "Criteo Terabyte",
                "split": "unshuffled-day-23-accuracy-set",
                "license_gate": "manual upstream terms acceptance",
                "redistributed_by_mlperf_edu": False,
                "required_preprocessed_files": [
                    "day_day_count.npz",
                    "day_fea_count.npz",
                    *[f"day_{day}_reordered.npz" for day in range(24)],
                ],
            },
            "checkpoint": {
                "name": "tb00_40M.pt",
                "url": DLRM_CHECKPOINT_URL,
                "md5": DLRM_CHECKPOINT_MD5,
            },
        },
        "source": {
            "inference_revision": DLRM_INFERENCE_COMMIT,
            "implementation_revision": DLRM_IMPLEMENTATION_COMMIT,
            "training_submodule_revision": DLRM_TRAINING_SUBMODULE_COMMIT,
        },
        "environment": {
            "MLPERF_EDU_CRITEO_TERMS_ACCEPTED": (
                "1 after reviewing and accepting the upstream terms"
            ),
            "MLPERF_EDU_DLRM_DATA_DIR": (
                "absolute directory containing all 26 preprocessed files"
            ),
            "MLPERF_EDU_DLRM_CHECKPOINT": (
                "absolute path to the MD5-pinned tb00_40M.pt"
            ),
            "MLPERF_EDU_DLRM_PYTHON": (
                "Python with torch, scikit-learn, and mlperf_loadgen"
            ),
            "MLPERF_EDU_DLRM_DEVICE": "cpu or gpu",
        },
        "preflight_command": (
            "mlperf doctor --workload recommendation --profile max --format json"
        ),
        "run_command": (
            "mlperf run --workload recommendation --profile max "
            "--output-dir submissions/recommendation-max"
        ),
        "production_ready": False,
        "remaining_after_quality": [
            "independent platform reproduction",
            "security and license review",
            "artifact signing",
            "later timing stability campaign",
        ],
    }


def minigo_environment_handoff_contract() -> dict[str, Any]:
    """Return the portable environment contract for MiniGo quality execution."""
    return {
        "schema": "mlperf-edu-environment-handoff/0.1",
        "workload": "reinforcement-learning",
        "profile": "max",
        "execution_status": "environment-gated-quality-conformance",
        "quality": {
            "metric": "professional_move_prediction",
            "target": 0.40,
            "direction": "higher",
            "acceptance_runs": 1,
            "secondary_gate": {
                "metric": "playoff_win_rate",
                "target": 0.55,
                "direction": "higher",
            },
        },
        "required_hardware": {
            "system": "single-node",
            "accelerator": "NVIDIA GPU",
            "container_gpu_interface": "Docker-compatible --gpus all",
            "legacy_runtime": "CUDA with the pinned TensorFlow 1.x environment",
        },
        "external_assets": {
            "professional_games": {
                "count": 4,
                "source_revision": MINIGO_COMMIT,
                "review_gate": "release and terms review before use",
            },
            "container_image": {
                "identity": "repository/image@sha256:<64 hex>",
                "build_source": "reinforcement/tensorflow/Dockerfile",
                "immutable_digest_required": True,
            },
            "self_play": {
                "generated_by_run": True,
                "games_per_generation": 2_000,
                "workers": 16,
                "search_readouts": 200,
            },
        },
        "source": {
            "training_revision": MINIGO_COMMIT,
            "critical_files": {
                name: f"sha256:{digest}" for name, digest in MINIGO_SOURCE_FILES.items()
            },
        },
        "environment": {
            "MLPERF_EDU_MINIGO_PRO_GAMES_REVIEWED": (
                "1 after release and terms review"
            ),
            "MLPERF_EDU_MINIGO_IMAGE": (
                "immutable repository/image@sha256:<64 hex>"
            ),
            "MLPERF_EDU_MINIGO_CONTAINER_RUNTIME": (
                "Docker-compatible executable; defaults to docker"
            ),
        },
        "preflight_command": (
            "mlperf doctor --workload reinforcement-learning --profile max "
            "--format json"
        ),
        "run_command": (
            "mlperf run --workload reinforcement-learning --profile max "
            "--output-dir submissions/reinforcement-learning-max"
        ),
        "resumable": True,
        "production_ready": False,
        "remaining_after_quality": [
            "independent platform reproduction",
            "security and professional-game terms review",
            "artifact signing",
            "later timing stability campaign",
        ],
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
    "mlperf-tiny-anomaly-eval": AssetDossier(
        id="mlperf-tiny-anomaly-eval",
        asset_type="dataset",
        display_name="MLPerf Tiny ToyCar anomaly-detection accuracy set",
        source_url=MLPERF_TINY_ANOMALY_ARCHIVE_URL,
        citation="Koizumi et al., ToyADMOS, 2019; DCASE 2020 Task 2; MLCommons MLPerf Tiny anomaly detection.",
        license="ToyADMOS Creative Commons Attribution 4.0 with MLCommons reference-code terms",
        license_spdx="CC-BY-4.0",
        license_status="cc-by-4.0-mlcommons-attribution",
        terms_summary="The pinned MLCommons index selects 248 ToyCar recordings from the immutable Zenodo archive and applies the upstream 128-bin librosa conversion recipe.",
        public_result_use="direct MLPerf Tiny score-bearing candidate with the fixed 0.85 ROC AUC gate",
        public_release_status="public-ok-fetch-only",
        public_release_policy="Fetch only the indexed recordings from Zenodo, preserve ToyADMOS and MLCommons attribution, and do not package the derived feature files.",
        release_next_step="Keep the Zenodo record, source-archive MD5, selected-member manifest, preprocessing versions, and derived-set hash in public artifacts.",
        license_evidence_url="https://zenodo.org/records/3678171",
        attribution="ToyADMOS, DCASE, Hitachi, and MLCommons contributors.",
        version=f"mlcommons-tiny-{MLPERF_TINY_ANOMALY_COMMIT}-toyadmos-3678171",
        expected_download_bytes=69_897_209,
        expected_unpacked_bytes=MLPERF_TINY_ANOMALY_DATASET_BYTES,
        hash_policy="The source archive is bound by Zenodo MD5, every selected ZIP member by its central-directory CRC and size, and the complete derived accuracy set by SHA-256.",
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
        license="Open Data Commons Attribution License 1.0 (ODC-By 1.0)",
        license_spdx="ODC-By-1.0",
        license_status="odc-by-1.0-attribution",
        terms_summary="The official OGB loader supplies 169,343 papers, 1,166,243 citation edges, 128-dimensional features, 40 classes, and the time-based split.",
        public_result_use="official OGB node-classification performance candidate",
        public_release_status="public-ok-fetch-only",
        public_release_policy="Fetch the pinned official archive, preserve OGB and Microsoft Academic Graph attribution, and do not bundle dataset bytes in portable result packages.",
        release_next_step="Keep the attribution notice and pinned source URL with every public result.",
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
    "humaneval-plus": AssetDossier(
        id="humaneval-plus",
        asset_type="dataset",
        display_name="EvalPlus HumanEval+ v0.1.10",
        source_url=HUMANEVAL_PLUS_URL,
        citation="Liu et al., Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation, NeurIPS 2023.",
        license="EvalPlus is Apache-2.0; HumanEval-derived benchmark components retain their upstream terms",
        license_spdx=None,
        license_status="apache-2.0-component-review-required",
        terms_summary="The pinned release contains all 164 HumanEval+ tasks and strengthened tests. Generated code must be evaluated in the declared sandbox.",
        public_result_use="Qwen2.5-Coder quality candidate with sandboxed EvalPlus execution",
        public_release_status="public-ok-fetch-only",
        public_release_policy="Fetch the pinned release at run time, preserve EvalPlus and HumanEval attribution, and never execute generated code outside the declared sandbox.",
        release_next_step="Keep the release version, compressed and uncompressed hashes, evaluator revision, and sandbox policy in public artifacts.",
        license_evidence_url="https://github.com/evalplus/evalplus/blob/master/LICENSE",
        attribution="EvalPlus and OpenAI HumanEval contributors.",
        version=HUMANEVAL_PLUS_VERSION,
        expected_download_bytes=925_000,
        expected_unpacked_bytes=HUMANEVAL_PLUS_BYTES,
        hash_policy="The compressed release and exact 164-record uncompressed JSONL are both pinned by SHA-256.",
    ),
    "bfcl-v4-non-live-ast": AssetDossier(
        id="bfcl-v4-non-live-ast",
        asset_type="dataset",
        display_name="BFCL V4 Non-Live AST",
        source_url=BFCL_ARCHIVE_URL,
        citation="Patil et al., Berkeley Function Calling Leaderboard; BFCL V4 official leaderboard and evaluator.",
        license="BFCL code and component dataset terms require release review",
        license_spdx=None,
        license_status="upstream-terms-review-required",
        terms_summary="The pinned source supplies all 1,150 examples in the six Non-Live AST categories, their possible answers, and the official AST evaluator.",
        public_result_use="Qwen3-1.7B function-calling quality candidate after component review",
        public_release_status="needs-release-decision",
        public_release_policy="Fetch the pinned source at run time and do not redistribute BFCL data until component terms are resolved.",
        release_next_step="Record the release decision and preserve the source, evaluator, category-file hashes, and official aggregation rule.",
        license_evidence_url="https://github.com/ShishirPatil/gorilla",
        attribution="Berkeley Gorilla and BFCL contributors.",
        version=f"bfcl-{BFCL_COMMIT}",
        expected_download_bytes=45_100_000,
        hash_policy="The GitHub source archive and the twelve question and possible-answer files are pinned by SHA-256.",
    ),
    "movielens-20m": AssetDossier(
        id="movielens-20m",
        asset_type="dataset",
        display_name="MovieLens 20M",
        source_url="https://files.grouplens.org/datasets/movielens/ml-20m.zip",
        citation="Harper and Konstan, The MovieLens Datasets: History and Context, ACM TiiS 2015.",
        license="GroupLens research terms permit research use with citation and prohibit redistribution without permission",
        license_spdx=None,
        license_status="research-use-with-citation",
        terms_summary="MLPerf Training v0.5 trains NCF on the 20-million-rating implicit-feedback set, holding out each user's last interaction and scoring it against 999 sampled negatives.",
        public_result_use="MLPerf Training v0.5 recommendation quality candidate",
        public_release_status="fetch-only",
        public_release_policy="Fetch the pinned archive from GroupLens. Do not redistribute the data inside result packages.",
        release_next_step="Record the GroupLens citation requirement in any published result package.",
        license_evidence_url="https://grouplens.org/datasets/movielens/20m/",
        attribution="GroupLens Research, University of Minnesota.",
        version="ml-20m",
        expected_download_bytes=198_702_078,
        hash_policy="Pin the archive SHA-256 and derive the ratings CSV from it rather than hashing extracted files.",
    ),
    "minigo-self-play": AssetDossier(
        id="minigo-self-play",
        asset_type="generated-dataset",
        display_name="MLPerf Training v0.5 MiniGo self-play stream",
        source_url=f"https://github.com/mlcommons/training/tree/{MINIGO_COMMIT}/reinforcement/minigo",
        citation="Mattson et al., MLPerf Training Benchmark, 2020; MLPerf Training v0.5 MiniGo.",
        license="Apache-2.0 reference code; generated games and professional-move inputs require release review",
        license_spdx=None,
        license_status="generated-and-upstream-input-review-required",
        terms_summary="Self-play games are generated during the run. The pinned source contains the four professional 9-by-9 SGF games used for move prediction and the 100-game promotion playoff.",
        public_result_use="historical MLPerf Training quality candidate after local feasibility review",
        public_release_status="needs-release-decision",
        public_release_policy="Fetch the pinned reference code and required inputs, retain generated games locally, and preserve the complete run recipe and hashes.",
        release_next_step="Resolve the professional-move input terms and document a practical authoritative execution environment.",
        license_evidence_url="https://github.com/mlcommons/training/blob/master/LICENSE.md",
        attribution="MLCommons and MiniGo contributors.",
        version=f"mlperf-training-{MINIGO_COMMIT}",
        expected_download_bytes=MINIGO_ARCHIVE_BYTES,
        hash_policy="Hash the pinned reference source, professional-move inputs, generated self-play stream, checkpoints, and playoff records.",
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


def _source_project_root() -> Path | None:
    from .registry import find_project_root

    root = find_project_root()
    return root if (root / "workloads.yaml").is_file() else None


def asset_cache_root() -> Path:
    """Return the stable root for benchmark-managed dataset assets."""
    override = os.environ.get("MLPERF_EDU_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    source_root = _source_project_root()
    if source_root is not None:
        return source_root / "data"
    return user_cache_path("mlperf-edu").resolve()


def data_root() -> Path:
    """Return the Tiny Shakespeare cache while preserving source-tree layout."""
    override = os.environ.get("MLPERF_EDU_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    source_root = _source_project_root()
    if source_root is not None:
        return source_root / "datasets" / "local_tensors"
    return asset_cache_root() / "tinyshakespeare"


def _asset_path_root(root: Path | None, name: str) -> Path:
    return root.resolve() if root is not None else asset_cache_root() / name


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
    base = _asset_path_root(root, "cifar10")
    return {
        "root": base,
        "dataset": base / "cifar-10-batches-py",
        "tar": base / "cifar-10-python.tar.gz",
    }


def fashion_mnist_paths(root: Path | None = None) -> dict[str, Path]:
    base = _asset_path_root(root, "fashion-mnist")
    return {
        "root": base,
        "raw": base / "FashionMNIST" / "raw",
        "processed": base / "FashionMNIST" / "processed",
    }


def mlperf_tiny_kws_paths(root: Path | None = None) -> dict[str, Path]:
    base = _asset_path_root(root, "mlperf-tiny-kws")
    return {
        "root": base,
        "dataset": base / "kws01",
        "archive": base / f"eembc-runner-{EEMBC_RUNNER_COMMIT}.tar.gz",
        "float_model": base / "kws_ref_model_float32.tflite",
        "int8_model": base / "kws_ref_model.tflite",
    }


def mlperf_tiny_vww_paths(root: Path | None = None) -> dict[str, Path]:
    base = _asset_path_root(root, "mlperf-tiny-vww")
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


def mlperf_tiny_anomaly_paths(root: Path | None = None) -> dict[str, Path]:
    base = _asset_path_root(root, "mlperf-tiny-anomaly")
    return {
        "root": base,
        "dataset": base / "ad01",
        "labels": base / "ad01" / "y_labels.csv",
        "float_model": base / "ad01_fp32.tflite",
        "int8_model": base / "ad01_int8.tflite",
    }


def mlperf_tiny_image_paths(root: Path | None = None) -> dict[str, Path]:
    base = _asset_path_root(root, "mlperf-tiny-image")
    return {
        "root": base,
        "float_model": base / "pretrainedResnet.tflite",
        "performance_indices": base / "perf_samples_idxs.npy",
        "evaluator_source": base / "eval_functions_eembc.py",
    }


def sst2_paths(root: Path | None = None) -> dict[str, Path]:
    base = _asset_path_root(root, "sst2")
    return {
        "root": base,
        "dataset": base / "SST-2",
        "zip": base / "SST-2.zip",
        "train": base / "SST-2" / "train.tsv",
        "validation": base / "SST-2" / "dev.tsv",
    }


def ogbn_arxiv_paths(root: Path | None = None) -> dict[str, Path]:
    base = _asset_path_root(root, "ogb")
    return {
        "root": base,
        "dataset": base / "ogbn_arxiv",
        "zip": base / "ogbn-arxiv-v1.zip",
    }


def ettm1_paths(root: Path | None = None) -> dict[str, Path]:
    base = _asset_path_root(root, "ettm1")
    return {"root": base, "csv": base / "ETTm1.csv"}


def nanobeir_reranking_paths(root: Path | None = None) -> dict[str, Path]:
    base = _asset_path_root(root, "nanobeir-reranking")
    return {"root": base}


def humaneval_plus_paths(root: Path | None = None) -> dict[str, Path]:
    base = _asset_path_root(root, "humaneval-plus")
    return {
        "root": base,
        "archive": base / "HumanEvalPlus-v0.1.10.jsonl.gz",
        "dataset": base / "HumanEvalPlus-v0.1.10.jsonl",
    }


def evalplus_evaluator_paths(root: Path | None = None) -> dict[str, Path]:
    base = _asset_path_root(root, "evalplus-evaluator")
    return {
        "root": base,
        "archive": base / f"evalplus-{EVALPLUS_COMMIT}.tar.gz",
        "source": base / f"evalplus-{EVALPLUS_COMMIT}",
    }


def bfcl_non_live_ast_paths(root: Path | None = None) -> dict[str, Path]:
    base = _asset_path_root(root, "bfcl-v4-non-live-ast")
    return {
        "root": base,
        "archive": base / f"gorilla-{BFCL_COMMIT}.tar.gz",
        "source": base / "berkeley-function-call-leaderboard",
        "data": base / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data",
    }


def edm_cifar10_paths(root: Path | None = None) -> dict[str, Path]:
    base = _asset_path_root(root, "edm-cifar10")
    return {
        "root": base,
        "checkpoint": base / "edm-cifar10-32x32-cond-vp.pkl",
        "fid_reference": base / "cifar10-32x32.npz",
        "inception": base / "inception-2015-12-05.pkl",
        "archive": base / f"edm-{EDM_COMMIT}.tar.gz",
        "source": base / f"edm-{EDM_COMMIT}",
    }


def dlrm_reference_paths(root: Path | None = None) -> dict[str, Path]:
    base = _asset_path_root(root, "dlrm-v1.0.1-reference")
    return {
        "root": base,
        "inference_archive": base / f"inference-{DLRM_INFERENCE_COMMIT}.tar.gz",
        "inference_source": base / f"inference-{DLRM_INFERENCE_COMMIT}",
        "implementation_archive": base / f"dlrm-{DLRM_IMPLEMENTATION_COMMIT}.tar.gz",
        "implementation_source": base / f"dlrm-{DLRM_IMPLEMENTATION_COMMIT}",
    }


def minigo_reference_paths(root: Path | None = None) -> dict[str, Path]:
    base = _asset_path_root(root, "minigo-v0.5-reference")
    return {
        "root": base,
        "archive": base / f"training-{MINIGO_COMMIT}.tar.gz",
        "source": base / f"training-{MINIGO_COMMIT}",
        "tensorflow": base
        / f"training-{MINIGO_COMMIT}"
        / "reinforcement"
        / "tensorflow",
        "minigo": base
        / f"training-{MINIGO_COMMIT}"
        / "reinforcement"
        / "tensorflow"
        / "minigo",
    }


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


def ensure_mlperf_tiny_anomaly(
    *, download: bool = True, root: Path | None = None
) -> DatasetAsset:
    """Prepare the complete 248-recording MLPerf Tiny ToyCar accuracy set."""
    paths = mlperf_tiny_anomaly_paths(root)
    base = paths["root"]
    dataset = paths["dataset"]
    base.mkdir(parents=True, exist_ok=True)

    current_files = tuple(sorted(path for path in dataset.rglob("*") if path.is_file()))
    current_digest = _dataset_file_digest(dataset, current_files)
    if (
        len(current_files) != 249
        or sum(path.stat().st_size for path in current_files)
        != MLPERF_TINY_ANOMALY_DATASET_BYTES
        or current_digest != MLPERF_TINY_ANOMALY_DATASET_SHA256
    ):
        if not download:
            raise FileNotFoundError(
                f"MLPerf Tiny anomaly data is missing at {dataset}. Run `mlperf "
                "fetch --workload anomaly-detection --profile max`."
            )

        import librosa
        import numpy as np
        import soundfile as sf
        from remotezip import RemoteZip

        staging = base / "ad01.staging"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        labels_path = staging / "y_labels.csv"
        _download(MLPERF_TINY_ANOMALY_LABELS_URL, labels_path)
        if sha256_file(labels_path) != MLPERF_TINY_ANOMALY_LABELS_SHA256:
            raise ValueError("MLPerf Tiny anomaly label index does not match its pin")

        with labels_path.open(newline="") as handle:
            rows = list(csv.reader(handle))
        labels = [int(row[2]) for row in rows if len(row) == 5]
        if (
            len(rows) != 248
            or len(labels) != 248
            or labels.count(0) != 140
            or labels.count(1) != 108
            or any(
                row[1:] not in [["2", "0", "2560", "512"], ["2", "1", "2560", "512"]]
                for row in rows
            )
        ):
            raise ValueError("MLPerf Tiny anomaly label index has an invalid contract")

        member_names = [
            "ToyCar/test/" + row[0].removesuffix("_hist_librosa.bin") + ".wav"
            for row in rows
        ]
        with RemoteZip(MLPERF_TINY_ANOMALY_ARCHIVE_URL) as archive:
            archive_members = {member.filename: member for member in archive.infolist()}
            member_digest = hashlib.sha256()
            for member_name in member_names:
                try:
                    member = archive_members[member_name]
                except KeyError as exc:
                    raise FileNotFoundError(
                        f"ToyADMOS archive is missing {member_name}"
                    ) from exc
                member_digest.update(
                    (
                        f"{member_name}\0{member.CRC:08x}\0{member.file_size}\0"
                        f"{member.compress_size}\n"
                    ).encode("utf-8")
                )
            if member_digest.hexdigest() != MLPERF_TINY_ANOMALY_MEMBER_MANIFEST_SHA256:
                raise ValueError(
                    "ToyADMOS selected-member manifest does not match its pin"
                )

            for index, (row, member_name) in enumerate(
                zip(rows, member_names, strict=True)
            ):
                audio_bytes = archive.read(member_name)
                audio, sample_rate = sf.read(
                    io.BytesIO(audio_bytes), dtype="float32", always_2d=False
                )
                if sample_rate != 16_000 or audio.shape != (176_000,):
                    raise ValueError(
                        f"unexpected ToyADMOS audio contract for {member_name}"
                    )
                mel_spectrogram = librosa.feature.melspectrogram(
                    y=audio,
                    sr=sample_rate,
                    n_fft=1024,
                    hop_length=512,
                    n_mels=128,
                    power=2.0,
                )
                log_mel = 10.0 * np.log10(mel_spectrogram + sys.float_info.epsilon)
                central = log_mel[:, 50:250]
                if central.shape != (128, 200):
                    raise ValueError(
                        f"unexpected ToyADMOS feature shape for {member_name}: "
                        f"{central.shape}"
                    )
                np.swapaxes(central, 0, 1).astype("<f4").tofile(staging / row[0])
                if index + 1 < len(rows):
                    time.sleep(0.5)

        staged_files = tuple(
            sorted(path for path in staging.rglob("*") if path.is_file())
        )
        staged_bytes = sum(path.stat().st_size for path in staged_files)
        staged_digest = _dataset_file_digest(staging, staged_files)
        if (
            len(staged_files) != 249
            or staged_bytes != MLPERF_TINY_ANOMALY_DATASET_BYTES
            or staged_digest != MLPERF_TINY_ANOMALY_DATASET_SHA256
        ):
            raise ValueError(
                "prepared MLPerf Tiny anomaly dataset does not match its pin"
            )
        shutil.rmtree(dataset, ignore_errors=True)
        staging.replace(dataset)

    model_specs = (
        (
            paths["float_model"],
            f"{MLPERF_TINY_ANOMALY_MODEL_BASE_URL}/ad01_fp32.tflite",
            MLPERF_TINY_ANOMALY_FLOAT_MODEL_SHA256,
        ),
        (
            paths["int8_model"],
            f"{MLPERF_TINY_ANOMALY_MODEL_BASE_URL}/ad01_int8.tflite",
            MLPERF_TINY_ANOMALY_INT8_MODEL_SHA256,
        ),
    )
    for model_path, url, expected_sha256 in model_specs:
        if not model_path.is_file() or sha256_file(model_path) != expected_sha256:
            if not download:
                raise FileNotFoundError(
                    f"Pinned MLPerf Tiny anomaly model is missing at {model_path}"
                )
            model_path.unlink(missing_ok=True)
            _download(url, model_path)
        if sha256_file(model_path) != expected_sha256:
            model_path.unlink(missing_ok=True)
            raise ValueError(
                f"MLPerf Tiny anomaly model SHA-256 mismatch: {model_path.name}"
            )

    files = tuple(sorted(path for path in dataset.rglob("*") if path.is_file()))
    return DatasetAsset(
        name="mlperf-tiny-anomaly-eval",
        root=dataset,
        files=files,
        sha256=f"sha256:{MLPERF_TINY_ANOMALY_DATASET_SHA256}",
        n_bytes=MLPERF_TINY_ANOMALY_DATASET_BYTES,
        source=MLPERF_TINY_ANOMALY_ARCHIVE_URL,
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
        (
            paths["evaluator_source"],
            f"{MLPERF_TINY_IMAGE_BASE_URL}/eval_functions_eembc.py",
            MLPERF_TINY_IMAGE_EVALUATOR_SHA256,
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


MOVIELENS_20M_URL = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"
MOVIELENS_20M_SHA256 = (
    "96f243c338a8665f6bcc89c53edf6ee39162a846940de6b7c8c48aeada765ff3"
)


def movielens_20m_paths(root: Path | None = None) -> dict[str, Path]:
    base = (root or asset_cache_root()) / "movielens-20m"
    return {
        "root": base,
        "archive": base / "ml-20m.zip",
        "ratings": base / "ml-20m" / "ratings.csv",
    }


def ensure_movielens_20m(
    *, download: bool = True, root: Path | None = None
) -> DatasetAsset:
    """Fetch and verify the official MovieLens-20M archive.

    The MLPerf Training v0.5 recommendation benchmark is defined on this exact
    release, so the archive digest is pinned rather than the extracted files.
    """
    paths = movielens_20m_paths(root)
    base = paths["root"]
    archive = paths["archive"]
    ratings = paths["ratings"]
    base.mkdir(parents=True, exist_ok=True)

    if not archive.is_file() or sha256_file(archive) != MOVIELENS_20M_SHA256:
        if not download:
            raise FileNotFoundError(
                f"MovieLens-20M is missing at {archive}. "
                "Run `mlperf fetch --workload recommendation --profile max`."
            )
        tmp = archive.with_suffix(".download")
        _download(MOVIELENS_20M_URL, tmp)
        if sha256_file(tmp) != MOVIELENS_20M_SHA256:
            tmp.unlink(missing_ok=True)
            raise ValueError("MovieLens-20M archive SHA-256 does not match the pinned value")
        tmp.replace(archive)

    if not ratings.is_file():
        import zipfile

        with zipfile.ZipFile(archive) as handle:
            handle.extract("ml-20m/ratings.csv", path=base)
    if not ratings.is_file():
        raise FileNotFoundError("MovieLens-20M ratings.csv was not extracted")

    return DatasetAsset(
        name="movielens-20m",
        root=base,
        files=(archive,),
        sha256=f"sha256:{MOVIELENS_20M_SHA256}",
        n_bytes=archive.stat().st_size,
        source=MOVIELENS_20M_URL,
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


def ensure_humaneval_plus(
    *, download: bool = True, root: Path | None = None
) -> DatasetAsset:
    paths = humaneval_plus_paths(root)
    base = paths["root"]
    archive = paths["archive"]
    dataset = paths["dataset"]
    base.mkdir(parents=True, exist_ok=True)

    archive_valid = (
        archive.is_file() and sha256_file(archive) == HUMANEVAL_PLUS_ARCHIVE_SHA256
    )
    dataset_valid = (
        dataset.is_file()
        and dataset.stat().st_size == HUMANEVAL_PLUS_BYTES
        and sha256_file(dataset) == HUMANEVAL_PLUS_SHA256
    )
    if not archive_valid or not dataset_valid:
        if not download:
            raise FileNotFoundError(
                f"HumanEval+ v0.1.10 is missing at {base}. Run `mlperf fetch "
                "--workload code-generation --profile max`."
            )
        if not archive_valid:
            archive.unlink(missing_ok=True)
            _download(HUMANEVAL_PLUS_URL, archive)
        if sha256_file(archive) != HUMANEVAL_PLUS_ARCHIVE_SHA256:
            archive.unlink(missing_ok=True)
            raise ValueError("HumanEval+ archive SHA-256 does not match the pin")
        with gzip.open(archive, "rb") as source, dataset.open("wb") as destination:
            shutil.copyfileobj(source, destination)

    if (
        dataset.stat().st_size != HUMANEVAL_PLUS_BYTES
        or sha256_file(dataset) != HUMANEVAL_PLUS_SHA256
    ):
        dataset.unlink(missing_ok=True)
        raise ValueError("HumanEval+ uncompressed JSONL does not match the pin")
    records = [json.loads(line) for line in dataset.read_text().splitlines()]
    task_ids = [record.get("task_id") for record in records]
    if len(records) != 164 or len(set(task_ids)) != 164 or None in task_ids:
        raise ValueError("HumanEval+ must contain 164 uniquely identified tasks")

    return DatasetAsset(
        name="humaneval-plus",
        root=base,
        files=(archive, dataset),
        sha256=f"sha256:{HUMANEVAL_PLUS_SHA256}",
        n_bytes=archive.stat().st_size + dataset.stat().st_size,
        source=HUMANEVAL_PLUS_URL,
    )


def ensure_evalplus_evaluator(
    *, download: bool = True, root: Path | None = None
) -> DatasetAsset:
    paths = evalplus_evaluator_paths(root)
    base = paths["root"]
    archive = paths["archive"]
    source_root = paths["source"]
    base.mkdir(parents=True, exist_ok=True)

    archive_valid = (
        archive.is_file() and sha256_file(archive) == EVALPLUS_ARCHIVE_SHA256
    )
    required = (source_root / "Dockerfile", source_root / "evalplus" / "evaluate.py")
    if not archive_valid or not all(path.is_file() for path in required):
        if not download:
            raise FileNotFoundError(
                f"EvalPlus evaluator source is missing at {source_root}. Run `mlperf "
                "fetch --workload code-generation --profile max`."
            )
        if not archive_valid:
            archive.unlink(missing_ok=True)
            _download(EVALPLUS_ARCHIVE_URL, archive)
        if sha256_file(archive) != EVALPLUS_ARCHIVE_SHA256:
            archive.unlink(missing_ok=True)
            raise ValueError("EvalPlus source archive SHA-256 does not match the pin")
        staging = base / f"evalplus-{EVALPLUS_COMMIT}.staging"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        prefix = f"evalplus-{EVALPLUS_COMMIT}/"
        with tarfile.open(archive, "r:gz") as bundle:
            for member in bundle.getmembers():
                if not member.isfile() or not member.name.startswith(prefix):
                    continue
                relative = Path(member.name.removeprefix(prefix))
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"unsafe EvalPlus archive member: {member.name}")
                source = bundle.extractfile(member)
                if source is None:
                    raise FileNotFoundError(
                        f"could not read EvalPlus archive member: {member.name}"
                    )
                destination = staging / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                with source, destination.open("wb") as handle:
                    shutil.copyfileobj(source, handle)
        shutil.rmtree(source_root, ignore_errors=True)
        staging.replace(source_root)

    if not all(path.is_file() for path in required):
        raise FileNotFoundError("EvalPlus pinned evaluator source is incomplete")
    return DatasetAsset(
        name="evalplus-evaluator",
        root=source_root,
        files=(archive, *required),
        sha256=f"sha256:{EVALPLUS_ARCHIVE_SHA256}",
        n_bytes=archive.stat().st_size,
        source=EVALPLUS_ARCHIVE_URL,
    )


def ensure_bfcl_non_live_ast(
    *, download: bool = True, root: Path | None = None
) -> DatasetAsset:
    paths = bfcl_non_live_ast_paths(root)
    base = paths["root"]
    archive = paths["archive"]
    source_root = paths["source"]
    data_root = paths["data"]
    base.mkdir(parents=True, exist_ok=True)

    selected = tuple(data_root / relative for relative in BFCL_DATA_FILES)
    evaluator_files = tuple(source_root / relative for relative in BFCL_EVALUATOR_FILES)
    selected_valid = all(
        path.is_file()
        and sha256_file(path) == BFCL_DATA_FILES[str(path.relative_to(data_root))]
        for path in selected
    )
    evaluator_valid = all(
        path.is_file()
        and sha256_file(path)
        == BFCL_EVALUATOR_FILES[str(path.relative_to(source_root))]
        for path in evaluator_files
    )
    if not selected_valid or not evaluator_valid:
        if not download:
            raise FileNotFoundError(
                f"BFCL V4 Non-Live AST is missing at {source_root}. Run `mlperf "
                "fetch --workload function-calling --profile max`."
            )
        if not archive.is_file() or sha256_file(archive) != BFCL_ARCHIVE_SHA256:
            archive.unlink(missing_ok=True)
            _download(BFCL_ARCHIVE_URL, archive)
        if sha256_file(archive) != BFCL_ARCHIVE_SHA256:
            archive.unlink(missing_ok=True)
            raise ValueError("BFCL source archive SHA-256 does not match the pin")

        prefix = f"gorilla-{BFCL_COMMIT}/berkeley-function-call-leaderboard/"
        staging = base / "berkeley-function-call-leaderboard.staging"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        with tarfile.open(archive, "r:gz") as bundle:
            for member in bundle.getmembers():
                if not member.isfile() or not member.name.startswith(prefix):
                    continue
                relative = Path(member.name.removeprefix(prefix))
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"unsafe BFCL archive member: {member.name}")
                source = bundle.extractfile(member)
                if source is None:
                    raise FileNotFoundError(
                        f"could not read BFCL archive member: {member.name}"
                    )
                destination = staging / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                with source, destination.open("wb") as handle:
                    shutil.copyfileobj(source, handle)
        shutil.rmtree(source_root, ignore_errors=True)
        staging.replace(source_root)

    selected = tuple(data_root / relative for relative in BFCL_DATA_FILES)
    for path in selected:
        relative = str(path.relative_to(data_root))
        if not path.is_file() or sha256_file(path) != BFCL_DATA_FILES[relative]:
            raise ValueError(f"BFCL data file does not match the pin: {relative}")

    evaluator_files = tuple(source_root / relative for relative in BFCL_EVALUATOR_FILES)
    for path in evaluator_files:
        relative = str(path.relative_to(source_root))
        if not path.is_file() or sha256_file(path) != BFCL_EVALUATOR_FILES[relative]:
            raise ValueError(f"BFCL evaluator file does not match the pin: {relative}")

    question_files = tuple(
        path
        for path in selected
        if "possible_answer" not in path.relative_to(data_root).parts
    )
    question_count = sum(
        1 for path in question_files for line in path.read_text().splitlines() if line
    )
    if question_count != 1150:
        raise ValueError(
            f"BFCL Non-Live AST expected 1150 examples, found {question_count}"
        )

    digest = hashlib.sha256()
    for path in selected:
        digest.update(str(path.relative_to(data_root)).encode("utf-8") + b"\0")
        digest.update(sha256_file(path).encode("ascii") + b"\n")
    return DatasetAsset(
        name="bfcl-v4-non-live-ast",
        root=data_root,
        files=(*selected, *evaluator_files),
        sha256=f"sha256:{digest.hexdigest()}",
        n_bytes=sum(path.stat().st_size for path in selected),
        source=BFCL_ARCHIVE_URL,
    )


def ensure_edm_cifar10(
    *, download: bool = True, root: Path | None = None
) -> DatasetAsset:
    paths = edm_cifar10_paths(root)
    base = paths["root"]
    base.mkdir(parents=True, exist_ok=True)
    assets = (
        (
            paths["checkpoint"],
            EDM_CIFAR10_CHECKPOINT_URL,
            EDM_CIFAR10_CHECKPOINT_SHA256,
        ),
        (
            paths["fid_reference"],
            EDM_CIFAR10_FID_REFERENCE_URL,
            EDM_CIFAR10_FID_REFERENCE_SHA256,
        ),
        (
            paths["inception"],
            EDM_INCEPTION_URL,
            EDM_INCEPTION_SHA256,
        ),
        (
            paths["archive"],
            EDM_ARCHIVE_URL,
            EDM_ARCHIVE_SHA256,
        ),
    )
    for path, url, expected_sha256 in assets:
        if not path.is_file() or sha256_file(path) != expected_sha256:
            if not download:
                raise FileNotFoundError(
                    f"EDM CIFAR-10 quality asset is missing at {path}. Run `mlperf "
                    "fetch --workload image-generation --profile max`."
                )
            path.unlink(missing_ok=True)
            _download(url, path)
        if sha256_file(path) != expected_sha256:
            path.unlink(missing_ok=True)
            raise ValueError(f"EDM CIFAR-10 SHA-256 mismatch: {path.name}")
    if paths["checkpoint"].stat().st_size != EDM_CIFAR10_CHECKPOINT_BYTES:
        raise ValueError("EDM CIFAR-10 checkpoint size does not match the pin")
    if paths["inception"].stat().st_size != EDM_INCEPTION_BYTES:
        raise ValueError("EDM Inception detector size does not match the pin")

    source_root = paths["source"]
    source_files = tuple(source_root / relative for relative in EDM_SOURCE_FILES)
    if not all(
        path.is_file()
        and sha256_file(path) == EDM_SOURCE_FILES[str(path.relative_to(source_root))]
        for path in source_files
    ):
        staging = base / f"edm-{EDM_COMMIT}.staging"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        prefix = f"edm-{EDM_COMMIT}/"
        with tarfile.open(paths["archive"], "r:gz") as bundle:
            for member in bundle.getmembers():
                if not member.isfile() or not member.name.startswith(prefix):
                    continue
                relative = Path(member.name.removeprefix(prefix))
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"unsafe EDM archive member: {member.name}")
                source = bundle.extractfile(member)
                if source is None:
                    raise FileNotFoundError(
                        f"could not read EDM archive member: {member.name}"
                    )
                destination = staging / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                with source, destination.open("wb") as handle:
                    shutil.copyfileobj(source, handle)
        shutil.rmtree(source_root, ignore_errors=True)
        staging.replace(source_root)
    source_files = tuple(source_root / relative for relative in EDM_SOURCE_FILES)
    for path in source_files:
        relative = str(path.relative_to(source_root))
        if not path.is_file() or sha256_file(path) != EDM_SOURCE_FILES[relative]:
            raise ValueError(f"EDM source file does not match the pin: {relative}")

    files = (
        paths["checkpoint"],
        paths["fid_reference"],
        paths["inception"],
        paths["archive"],
        *source_files,
    )
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8") + b"\0")
        digest.update(sha256_file(path).encode("ascii") + b"\n")
    return DatasetAsset(
        name="edm-cifar10-quality-assets",
        root=base,
        files=files,
        sha256=f"sha256:{digest.hexdigest()}",
        n_bytes=sum(path.stat().st_size for path in files),
        source=f"https://github.com/NVlabs/edm/tree/{EDM_COMMIT}",
    )


def _extract_pinned_source_archive(
    *,
    archive: Path,
    source_root: Path,
    prefix: str,
    expected_files: dict[str, str],
) -> tuple[Path, ...]:
    source_files = tuple(source_root / relative for relative in expected_files)
    if not all(
        path.is_file()
        and sha256_file(path) == expected_files[str(path.relative_to(source_root))]
        for path in source_files
    ):
        staging = source_root.with_name(f"{source_root.name}.staging")
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        with tarfile.open(archive, "r:gz") as bundle:
            for member in bundle.getmembers():
                if not member.isfile() or not member.name.startswith(prefix):
                    continue
                relative = Path(member.name.removeprefix(prefix))
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"unsafe source archive member: {member.name}")
                source = bundle.extractfile(member)
                if source is None:
                    raise FileNotFoundError(
                        f"could not read source archive member: {member.name}"
                    )
                destination = staging / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                with source, destination.open("wb") as handle:
                    shutil.copyfileobj(source, handle)
        shutil.rmtree(source_root, ignore_errors=True)
        staging.replace(source_root)

    source_files = tuple(source_root / relative for relative in expected_files)
    for path in source_files:
        relative = str(path.relative_to(source_root))
        if not path.is_file() or sha256_file(path) != expected_files[relative]:
            raise ValueError(f"pinned source file does not match: {relative}")
    return source_files


def ensure_dlrm_reference(
    *, download: bool = True, root: Path | None = None
) -> DatasetAsset:
    """Fetch the exact MLPerf Inference and DLRM implementation sources."""
    paths = dlrm_reference_paths(root)
    paths["root"].mkdir(parents=True, exist_ok=True)
    archives = (
        (
            paths["inference_archive"],
            DLRM_INFERENCE_ARCHIVE_URL,
            DLRM_INFERENCE_ARCHIVE_SHA256,
        ),
        (
            paths["implementation_archive"],
            DLRM_IMPLEMENTATION_ARCHIVE_URL,
            DLRM_IMPLEMENTATION_ARCHIVE_SHA256,
        ),
    )
    for archive, url, expected_sha256 in archives:
        if not archive.is_file() or sha256_file(archive) != expected_sha256:
            if not download:
                raise FileNotFoundError(
                    f"DLRM v1.0.1 reference source is missing at {archive}. Run "
                    "`mlperf fetch --workload recommendation --profile max`."
                )
            archive.unlink(missing_ok=True)
            _download(url, archive)
        if sha256_file(archive) != expected_sha256:
            archive.unlink(missing_ok=True)
            raise ValueError(f"DLRM reference SHA-256 mismatch: {archive.name}")

    inference_files = _extract_pinned_source_archive(
        archive=paths["inference_archive"],
        source_root=paths["inference_source"],
        prefix=f"inference-{DLRM_INFERENCE_COMMIT}/",
        expected_files=DLRM_INFERENCE_FILES,
    )
    implementation_files = _extract_pinned_source_archive(
        archive=paths["implementation_archive"],
        source_root=paths["implementation_source"],
        prefix=f"dlrm-{DLRM_IMPLEMENTATION_COMMIT}/",
        expected_files=DLRM_IMPLEMENTATION_FILES,
    )
    files = (
        paths["inference_archive"],
        paths["implementation_archive"],
        *inference_files,
        *implementation_files,
    )
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8") + b"\0")
        digest.update(sha256_file(path).encode("ascii") + b"\n")
    return DatasetAsset(
        name="mlperf-inference-v1.0.1-dlrm-reference",
        root=paths["root"],
        files=files,
        sha256=f"sha256:{digest.hexdigest()}",
        n_bytes=sum(path.stat().st_size for path in files),
        source=(
            f"https://github.com/mlcommons/inference/tree/{DLRM_INFERENCE_COMMIT}/"
            "recommendation/dlrm/pytorch"
        ),
    )


def ensure_minigo_reference(
    *, download: bool = True, root: Path | None = None
) -> DatasetAsset:
    """Fetch the exact historical MiniGo source and professional SGF inputs."""
    paths = minigo_reference_paths(root)
    paths["root"].mkdir(parents=True, exist_ok=True)
    archive = paths["archive"]
    if (
        not archive.is_file()
        or archive.stat().st_size != MINIGO_ARCHIVE_BYTES
        or sha256_file(archive) != MINIGO_ARCHIVE_SHA256
    ):
        if not download:
            raise FileNotFoundError(
                f"MiniGo v0.5 reference source is missing at {archive}. Run "
                "`mlperf fetch --workload reinforcement-learning --profile max`."
            )
        archive.unlink(missing_ok=True)
        _download(MINIGO_ARCHIVE_URL, archive)
    if (
        archive.stat().st_size != MINIGO_ARCHIVE_BYTES
        or sha256_file(archive) != MINIGO_ARCHIVE_SHA256
    ):
        archive.unlink(missing_ok=True)
        raise ValueError("MiniGo source archive does not match its pinned bytes")

    source_files = _extract_pinned_source_archive(
        archive=archive,
        source_root=paths["source"],
        prefix=f"training-{MINIGO_COMMIT}/",
        expected_files=MINIGO_SOURCE_FILES,
    )
    professional_games = tuple(
        path
        for path in source_files
        if "benchmark_sgf" in path.parts and path.suffix == ".sgf"
    )
    if len(professional_games) != 4:
        raise ValueError("MiniGo quality contract requires exactly four SGF games")

    digest = hashlib.sha256()
    files = (archive, *source_files)
    for path in files:
        digest.update(path.name.encode("utf-8") + b"\0")
        digest.update(sha256_file(path).encode("ascii") + b"\n")
    return DatasetAsset(
        name="mlperf-training-v0.5-minigo-reference",
        root=paths["source"],
        files=files,
        sha256=f"sha256:{digest.hexdigest()}",
        n_bytes=sum(path.stat().st_size for path in files),
        source=(
            f"https://github.com/mlcommons/training/tree/{MINIGO_COMMIT}/"
            "reinforcement/tensorflow"
        ),
    )


def _download(url: str, destination: Path) -> None:
    # Download to a process- and call-unique path first, then atomically move
    # it into place. Two concurrent fetches of the same asset (e.g. two
    # workload runs racing to populate a shared dataset cache) must not write
    # through the same temp file, or a slower writer's in-progress download
    # can be clobbered mid-write by the other process before either renames.
    unique_tmp = destination.with_name(
        f"{destination.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.part"
    )
    try:
        try:
            urllib.request.urlretrieve(url, unique_tmp)
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
                            str(unique_tmp),
                        ],
                        check=True,
                    )
                except subprocess.CalledProcessError as curl_exc:
                    raise RuntimeError(
                        f"failed to download {url} with urllib and curl"
                    ) from curl_exc
            else:
                raise RuntimeError(f"failed to download {url}") from urllib_exc
        unique_tmp.replace(destination)
    finally:
        if unique_tmp.exists():
            unique_tmp.unlink()
