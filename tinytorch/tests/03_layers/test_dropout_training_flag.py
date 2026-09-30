"""Dropout follows the layer's training flag, set by train()/eval() or the Trainer (2026-09-28).

Before this fix Dropout.forward only read its `training` argument (default True),
so a hand-written model evaluated by Trainer.evaluate kept masking at eval time.
"""
import numpy as np

from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Layer, Linear, Dropout, Sequential, set_training_mode
from tinytorch.core.activations import ReLU
from tinytorch.core.losses import MSELoss
from tinytorch.core.optimizers import SGD
from tinytorch.core.training import Trainer


class HandWritten:
    """A model that is a plain class, not a Layer or a Sequential."""

    def __init__(self):
        self.fc1 = Linear(8, 16)
        self.drop = Dropout(0.5)
        self.fc2 = Linear(16, 1)

    def forward(self, x):
        return self.fc2(self.drop(ReLU()(self.fc1(x))))

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        return self.fc1.parameters() + self.fc2.parameters()


def _data():
    x = Tensor(np.random.default_rng(0).normal(size=(64, 8)).astype(np.float32))
    y = Tensor(np.zeros((64, 1), dtype=np.float32))
    return x, y


def test_layers_default_to_training_mode():
    assert Dropout(0.5).training is True
    assert Linear(2, 2).training is True
    assert Sequential(Linear(2, 2)).training is True


def test_dropout_eval_flag_disables_mask_without_argument():
    x = Tensor(np.ones((4, 100), dtype=np.float32))
    d = Dropout(0.5)
    assert d.eval() is d
    assert d.training is False
    np.testing.assert_array_equal(d(x).data, x.data)
    np.testing.assert_array_equal(d.forward(x).data, x.data)
    d.train()
    assert (d(x).data == 0).any(), "a trained-mode Dropout(0.5) must mask something"


def test_explicit_argument_overrides_flag():
    x = Tensor(np.ones((4, 100), dtype=np.float32))
    d = Dropout(0.5).eval()
    assert (d.forward(x, training=True).data == 0).any()
    d.train()
    np.testing.assert_array_equal(d.forward(x, training=False).data, x.data)


def test_sequential_eval_reaches_every_layer():
    model = Sequential(Linear(4, 4), ReLU(), Dropout(0.5), Linear(4, 1))
    model.eval()
    assert model.training is False
    assert model.layers[2].training is False
    x = Tensor(np.ones((8, 4), dtype=np.float32))
    np.testing.assert_array_equal(model(x).data, model(x).data)
    model.train()
    assert model.layers[2].training is True


def test_set_training_mode_walks_nested_containers_and_cycles():
    class Block(Layer):
        def __init__(self):
            self.parts = [Linear(2, 2), Dropout(0.5)]
            self.named = {"drop": Dropout(0.5)}
            self.me = self  # circular reference must not recurse forever

        def forward(self, x):
            return x

    block = Block()
    outer = Sequential(block)
    set_training_mode(outer, False)
    assert block.training is False
    assert block.parts[1].training is False
    assert block.named["drop"].training is False
    block.train()
    assert block.parts[1].training is True
    assert block.named["drop"].training is True


def test_trainer_evaluate_is_deterministic_for_hand_written_model():
    model = HandWritten()
    trainer = Trainer(model, SGD(model.parameters(), lr=0.01), MSELoss())
    x, y = _data()
    losses = [trainer.evaluate([(x, y)])[0] for _ in range(5)]
    assert len(set(losses)) == 1, f"eval loss varied run to run: {losses}"
    assert model.drop.training is False

    trainer.train_epoch([(x, y)])
    assert model.drop.training is True, "train_epoch must switch dropout back on"
