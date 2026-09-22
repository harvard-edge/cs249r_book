"""Regressions against Module 08 source, independent of its generated export."""
from pathlib import Path
import pickle
import re
import types

import numpy as np
import pytest

from tinytorch.core.layers import Linear
from tinytorch.core.losses import BinaryCrossEntropyLoss, CrossEntropyLoss, MSELoss
from tinytorch.core.optimizers import SGD
from tinytorch.core.tensor import Tensor


@pytest.fixture(scope="module")
def training_source():
    path = Path(__file__).resolve().parents[2] / "src/08_training/08_training.py"
    module = types.ModuleType("training_source")
    for cell in re.split(r"^# %%.*$", path.read_text(encoding="utf-8"), flags=re.MULTILINE):
        if re.search(r"^#\| exporti?\s*$", cell, re.MULTILINE):
            exec(compile(cell, str(path), "exec"), module.__dict__)
    return module


class GeneratorLinear(Linear):
    def parameters(self):
        yield from super().parameters()


def make_trainer(source, model_type=Linear, clip=None):
    model = model_type(1, 1)
    model.weight.data[:] = 1.0
    model.bias.data[:] = 0.0
    return source.Trainer(model, SGD(model.parameters(), lr=0.1), MSELoss(),
                          grad_clip_norm=clip)


def test_generator_parameters_receive_gradient_clipping(training_source):
    trainer = make_trainer(training_source, GeneratorLinear, clip=0.5)
    trainer.train_epoch([(Tensor([[10.]]), Tensor([[0.]]))])
    # Unclipped gradient is [200, 20]; its direction must survive clipping.
    expected_update = 0.1 * 0.5 * np.array([200., 20.]) / np.linalg.norm([200., 20.])
    np.testing.assert_allclose(trainer.model.weight.data, [[1. - expected_update[0]]])
    np.testing.assert_allclose(trainer.model.bias.data, [-expected_update[1]])


@pytest.mark.parametrize("saved_clip,initial_clip", [(0.5, None), (None, 0.5)])
def test_checkpoint_restores_clipping_and_next_update(training_source, tmp_path,
                                                      saved_clip, initial_clip):
    original = make_trainer(training_source, clip=saved_clip)
    restored = make_trainer(training_source, clip=initial_clip)
    batches = [(Tensor([[10.]]), Tensor([[0.]]))]
    original.train_epoch(batches)
    path = tmp_path / "resume.pkl"
    original.save_checkpoint(path)
    restored.load_checkpoint(path)
    assert restored.grad_clip_norm == saved_clip
    expected_loss = original.train_epoch(batches)
    actual_loss = restored.train_epoch(batches)
    assert actual_loss == pytest.approx(expected_loss)
    for actual, expected in zip(restored.model.parameters(), original.model.parameters()):
        np.testing.assert_allclose(actual.data, expected.data)


def test_legacy_checkpoint_preserves_current_clipping(training_source, tmp_path):
    trainer = make_trainer(training_source, clip=0.5)
    path = tmp_path / "legacy.pkl"
    trainer.save_checkpoint(path)
    with path.open("rb") as stream:
        state = pickle.load(stream)
    del state["grad_clip_norm"]
    with path.open("wb") as stream:
        pickle.dump(state, stream)
    trainer.load_checkpoint(path)
    assert trainer.grad_clip_norm == 0.5


class IdentityModel:
    def parameters(self):
        return []

    def forward(self, x):
        return x


@pytest.mark.parametrize("loss,outputs,targets,accuracy", [
    (BinaryCrossEntropyLoss(), [[0.9], [0.1]], [[1.], [0.]], 1.0),
    (BinaryCrossEntropyLoss(), [0.5, 0.1, 0.9], [1., 0., 0.], 2 / 3),
    (BinaryCrossEntropyLoss(), [[0.9, 0.1], [0.9, 0.9]], [[1., 0.], [0., 1.]], 0.75),
    (CrossEntropyLoss(), [[3., 1.], [1., 3.], [3., 1.]], [0, 1, 1], 2 / 3),
    (MSELoss(), [[2., 3.], [4., 5.]], [[2., 3.], [4., 5.]], 0.0),
])
def test_accuracy_follows_loss_type_and_batch_partition(training_source, loss,
                                                       outputs, targets, accuracy):
    trainer = training_source.Trainer(IdentityModel(), SGD([]), loss)
    x, y = np.array(outputs), np.array(targets)
    whole_loss, whole_accuracy = trainer.evaluate([(Tensor(x), Tensor(y))])
    split_loss, split_accuracy = trainer.evaluate([
        (Tensor(x[:1]), Tensor(y[:1])), (Tensor(x[1:]), Tensor(y[1:]))])
    assert whole_accuracy == pytest.approx(accuracy)
    assert split_accuracy == pytest.approx(accuracy)
    assert split_loss == pytest.approx(whole_loss)


def test_custom_loss_has_no_inferred_classification_metric(training_source):
    class CustomLoss:
        def forward(self, outputs, targets):
            return MSELoss().forward(outputs, targets)

    trainer = training_source.Trainer(IdentityModel(), SGD([]), CustomLoss())
    loss, accuracy = trainer.evaluate([(Tensor([[1., 2.]]), Tensor([[1., 2.]]))])
    assert loss == 0.0
    assert accuracy == 0.0
