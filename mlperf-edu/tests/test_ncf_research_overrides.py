"""The NCF `pro`-profile research overrides.

These exist so the learning-rate-schedule hypothesis for the recommendation
shortfall can be tested without editing the canonical contract. The property
that matters most is the first one: an unset environment must reproduce the
contract exactly, or every ablation is measured against a moved baseline.
"""

from __future__ import annotations

import pytest
import torch

from mlperf.fingerprint import PERFORMANCE_ENVIRONMENT_ALLOWLIST
from mlperf.runners.ncf import (
    LR_SCHEDULES,
    _build_lr_scheduler,
    _lr_schedule_override,
    _optimizer_lr_override,
)

CONTRACT = {"learning_rate": 0.0005}


def test_unset_environment_reproduces_the_contract(monkeypatch):
    monkeypatch.delenv("MLPERF_EDU_NCF_LEARNING_RATE", raising=False)
    monkeypatch.delenv("MLPERF_EDU_NCF_LR_SCHEDULE", raising=False)
    assert _optimizer_lr_override(CONTRACT) == 0.0005
    assert _lr_schedule_override() == "constant"


def test_empty_string_is_treated_as_unset(monkeypatch):
    monkeypatch.setenv("MLPERF_EDU_NCF_LEARNING_RATE", "")
    monkeypatch.setenv("MLPERF_EDU_NCF_LR_SCHEDULE", "")
    assert _optimizer_lr_override(CONTRACT) == 0.0005
    assert _lr_schedule_override() == "constant"


def test_learning_rate_override_is_applied(monkeypatch):
    monkeypatch.setenv("MLPERF_EDU_NCF_LEARNING_RATE", "0.001")
    assert _optimizer_lr_override(CONTRACT) == pytest.approx(0.001)


@pytest.mark.parametrize("bad", ["abc", "0", "-0.001", "nan"])
def test_learning_rate_fails_closed_on_bad_values(monkeypatch, bad):
    monkeypatch.setenv("MLPERF_EDU_NCF_LEARNING_RATE", bad)
    with pytest.raises(ValueError):
        _optimizer_lr_override(CONTRACT)


@pytest.mark.parametrize("schedule", LR_SCHEDULES)
def test_every_declared_schedule_is_accepted(monkeypatch, schedule):
    monkeypatch.setenv("MLPERF_EDU_NCF_LR_SCHEDULE", schedule.upper())
    assert _lr_schedule_override() == schedule


def test_schedule_fails_closed_on_unknown_value(monkeypatch):
    monkeypatch.setenv("MLPERF_EDU_NCF_LR_SCHEDULE", "exponential")
    with pytest.raises(ValueError):
        _lr_schedule_override()


def test_constant_schedule_builds_no_scheduler():
    opt = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=0.0005)
    assert _build_lr_scheduler(opt, "constant", 7) is None


@pytest.mark.parametrize("schedule", ["cosine", "step"])
def test_annealing_schedules_lower_the_rate_over_the_budget(schedule):
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam([param], lr=0.0005)
    sched = _build_lr_scheduler(opt, schedule, 7)
    assert sched is not None
    first = opt.param_groups[0]["lr"]
    for _ in range(7):
        sched.step()
    assert opt.param_groups[0]["lr"] < first


def test_overrides_are_stamped_into_the_fingerprint():
    # An ablation that is not recorded in provenance is not evidence.
    assert "MLPERF_EDU_NCF_LEARNING_RATE" in PERFORMANCE_ENVIRONMENT_ALLOWLIST
    assert "MLPERF_EDU_NCF_LR_SCHEDULE" in PERFORMANCE_ENVIRONMENT_ALLOWLIST
