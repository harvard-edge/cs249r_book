#!/usr/bin/env python3
"""Train the running network on real MNIST and record overfitting (Lecture 5).

A real run, not a mock: the 784-128-64-10 ReLU network, FP32, batch 32,
Adam (learning rate 0.001), one CPU thread, fixed seed. Two training sets
drawn from the official MNIST training file: the first 1,000 images and all
60,000. After each epoch the script measures accuracy on the training set
and on the 10,000 held-out test images. Results go to data/overfit.json.

It also copies the first MNIST test image (its pixels and label) to
data/mnist_test0.json, so the slide figures draw a real digit.

MNIST: LeCun, Cortes, and Burges, yann.lecun.com/exdb/mnist; files fetched
from the torchvision mirror, ossci-datasets.s3.amazonaws.com/mnist/.
Set MNIST_DIR to a folder holding the four .gz files (downloaded if absent).

Needs torch (not in the MLSysBook venv; the system python3 has it):
  MNIST_DIR=/tmp/mnist python3 overfit_run.py
"""
import gzip
import json
import os
import platform
import struct
import urllib.request
from pathlib import Path

import torch
from torch import nn

MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
         "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
SRC = Path(os.environ.get("MNIST_DIR", "/tmp/mnist"))
DATA = Path(__file__).resolve().parent / "data"
EPOCHS = 40
SIZES = (1000, 60000)


def fetch():
    SRC.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        if not (SRC / f).exists():
            urllib.request.urlretrieve(MIRROR + f, SRC / f)


def images(name):
    raw = gzip.open(SRC / name).read()
    _, n, r, c = struct.unpack(">IIII", raw[:16])
    return torch.frombuffer(bytearray(raw[16:]), dtype=torch.uint8).reshape(n, r * c)


def labels(name):
    raw = gzip.open(SRC / name).read()
    return torch.frombuffer(bytearray(raw[8:]), dtype=torch.uint8).long()


def accuracy(net, x, y):
    with torch.no_grad():
        return (net(x).argmax(1) == y).float().mean().item()


def run(n, xtr, ytr, xte, yte):
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 64),
                        nn.ReLU(), nn.Linear(64, 10))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    x, y = xtr[:n], ytr[:n]
    hist = {"train_acc": [], "test_acc": []}
    for ep in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, 32):
            idx = perm[i:i + 32]
            loss = loss_fn(net(x[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        hist["train_acc"].append(accuracy(net, x, y))
        hist["test_acc"].append(accuracy(net, xte, yte))
        print(f"n={n} epoch {ep + 1}: train {hist['train_acc'][-1]:.4f} "
              f"held-out {hist['test_acc'][-1]:.4f}")
    return hist


def main():
    fetch()
    torch.set_num_threads(1)
    xtr = images(FILES[0]).float() / 255.0
    ytr = labels(FILES[1])
    xte_u8 = images(FILES[2])
    xte = xte_u8.float() / 255.0
    yte = labels(FILES[3])
    DATA.mkdir(exist_ok=True)
    (DATA / "mnist_test0.json").write_text(json.dumps(
        {"source": "MNIST t10k-images-idx3-ubyte, image 0",
         "label": int(yte[0]), "pixels": xte_u8[0].tolist()}))
    out = {"setup": f"784-128-64-10 ReLU, FP32, batch 32, Adam lr 1e-3, {EPOCHS} epochs, "
                    f"seed 0, 1 CPU thread; held-out = the 10,000 MNIST test images",
           "versions": f"torch {torch.__version__}; Python {platform.python_version()}; "
                       f"{platform.platform()}",
           "runs": {str(n): run(n, xtr, ytr, xte, yte) for n in SIZES}}
    (DATA / "overfit.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {DATA / 'overfit.json'}")


if __name__ == "__main__":
    main()
