import torch

from mlperf.runners.common import select_torch_device


def test_select_torch_device_honors_explicit_request(monkeypatch):
    monkeypatch.setenv("MLPERF_EDU_DEVICE", "cpu")

    assert select_torch_device() == torch.device("cpu")


def test_select_torch_device_auto_prefers_cuda(monkeypatch):
    monkeypatch.delenv("MLPERF_EDU_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert select_torch_device() == torch.device("cuda")


def test_select_torch_device_auto_uses_mps_before_cpu(monkeypatch):
    monkeypatch.delenv("MLPERF_EDU_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert select_torch_device() == torch.device("mps")


def test_select_torch_device_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.delenv("MLPERF_EDU_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    assert select_torch_device() == torch.device("cpu")
