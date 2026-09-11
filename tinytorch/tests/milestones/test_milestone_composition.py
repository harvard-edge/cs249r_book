"""Milestone composition must preserve gradients, parameters, and reported losses."""
import numpy as np
import pytest

from test_milestones_smoke import MILESTONES_DIR, _import_milestone
from tinytorch.core.tensor import Tensor
from tinytorch.core.losses import CrossEntropyLoss
from tinytorch.core.optimizers import Adam, SGD


def milestone(path):
    return _import_milestone(MILESTONES_DIR / path)


def test_tinydigits_cnn_trains_convolution_and_classifier():
    module = milestone('04_1998_cnn/01_lecun_tinydigits.py')
    model = module.SimpleCNN()
    params = model.parameters()
    optimizer = SGD(params, lr=0.03)
    rng = np.random.default_rng(42)
    images = Tensor(rng.normal(size=(8, 1, 8, 8)))
    labels = Tensor(np.arange(8) % 3)
    before = [p.data.copy() for p in params]
    losses = []
    for _ in range(8):
        optimizer.zero_grad()
        loss = CrossEntropyLoss()(model(images), labels)
        losses.append(float(loss.data))
        loss.backward()
        assert all(p.grad is not None and np.isfinite(p.grad).all() for p in params)
        optimizer.step()
    assert all(np.any(p.data != old) for p, old in zip(params, before))
    assert losses[-1] < losses[0]


def test_tinydigits_evaluation_reports_cross_entropy():
    module = milestone('04_1998_cnn/01_lecun_tinydigits.py')
    logits = Tensor([[3, 1, -2], [-1, 0, 2]])
    labels = Tensor([0, 1])
    accuracy, loss = module.evaluate_accuracy(lambda _: logits, None, labels)
    assert accuracy == 50
    expected = np.log(np.exp(logits.data).sum(axis=1)) - logits.data[[0, 1], [0, 1]]
    assert loss == pytest.approx(expected.mean())


def test_mlp_flatten_preserves_input_gradient():
    module = milestone('03_1986_mlp/01_rumelhart_tinydigits.py')
    model = module.DigitMLP(input_size=4, hidden_size=8, num_classes=3)
    SGD(model.parameters(), lr=0.01)
    images = Tensor(np.ones((2, 2, 2)), requires_grad=True)
    CrossEntropyLoss()(model(images), Tensor([0, 1])).backward()
    assert images.grad is not None
    assert np.any(images.grad != 0)


def test_transformer_registers_and_trains_each_parameter_group():
    module = milestone('05_2017_transformer/01_vaswani_attention.py')
    model = module.AttentionTransformer(7, embed_dim=8, num_heads=2, seq_len=4, num_layers=1)
    groups = [model.embedding, model.pos_encoding, model.attention_layers[0],
              model.ln1_layers[0], model.ln2_layers[0], model.fc1_layers[0],
              model.fc2_layers[0], model.output_proj]
    intended = [p for group in groups for p in group.parameters()]
    params = model.parameters()
    assert len({id(p) for p in params}) == len(params)
    assert {id(p) for p in params} == {id(p) for p in intended}
    optimizer = Adam(params, lr=0.01)
    inputs = Tensor([[1, 2, 3, 4], [4, 3, 2, 1], [2, 4, 1, 3]])
    targets = Tensor([[4, 3, 2, 1], [1, 2, 3, 4], [3, 1, 4, 2]])
    before = {id(p): p.data.copy() for p in params}
    losses = []
    for _ in range(12):
        optimizer.zero_grad()
        logits = model(inputs)
        loss = CrossEntropyLoss()(logits.reshape(-1, 7), targets.reshape(-1))
        losses.append(float(loss.data))
        loss.backward()
        assert all(p.grad is not None and np.isfinite(p.grad).all() for p in params)
        optimizer.step()
    # Key-projection bias can have zero gradient: adding the same key bias to
    # every position shifts each softmax row uniformly. Check every parameter
    # is registered/reached, and every functional layer actually changes.
    for group in groups:
        assert any(np.any(p.data != before[id(p)]) for p in group.parameters())
    assert all(np.any(p.data != before[id(p)]) for p in model.pos_encoding.parameters())
    assert losses[-1] < losses[0]


def test_cifar_architecture_check_does_not_download(monkeypatch):
    module = milestone('04_1998_cnn/02_lecun_cifar10.py')
    def unexpected_download(*args, **kwargs):
        pytest.fail('--test-only must not download CIFAR-10')
    monkeypatch.setattr(module.DatasetManager, 'get_cifar10', unexpected_download)
    monkeypatch.setattr('sys.argv', ['milestone', '--test-only'])
    module.main()


@pytest.mark.parametrize('count', [0, 3, 5, 101])
def test_xor_rejects_unbalanced_sample_count(count):
    module = milestone('02_1969_xor/02_xor_solved.py')
    with pytest.raises(ValueError, match='multiple of four'):
        module.generate_xor_data(count)


def test_xor_network_learns_all_four_cases(monkeypatch):
    import tinytorch.core.layers as layers
    module = milestone('02_1969_xor/02_xor_solved.py')
    monkeypatch.setattr(layers, 'rng', np.random.default_rng(1986))
    model = module.XORNetwork()
    images, labels = module.generate_xor_data(100)
    before = [p.data.copy() for p in model.parameters()]
    history = module.train_network(model, images, labels, epochs=500)
    predictions = model(Tensor([[0, 0], [0, 1], [1, 0], [1, 1]]))
    np.testing.assert_array_equal(predictions.data > 0.5, [[False], [True], [True], [False]])
    assert history['loss'][-1] < history['loss'][0]
    assert all(np.any(p.data != old) for p, old in zip(model.parameters(), before))


def test_tinydigits_epoch_loss_weights_partial_batch_by_samples():
    from tinytorch.core.dataloader import DataLoader, TensorDataset
    module = milestone('04_1998_cnn/01_lecun_tinydigits.py')
    model = module.SimpleCNN()
    optimizer = SGD(model.parameters(), lr=0.0)
    images = Tensor(np.random.default_rng(6).normal(size=(5, 1, 8, 8)))
    labels = Tensor([0, 1, 2, 3, 4])
    expected = float(CrossEntropyLoss()(model(images), labels).data)
    loader = DataLoader(TensorDataset(images, labels), batch_size=3, shuffle=False)
    actual = module.train_epoch(model, loader, CrossEntropyLoss(), optimizer)
    assert actual == pytest.approx(expected, rel=1e-6)


def test_perceptron_forward_matches_selected_weights():
    module = milestone('01_1958_perceptron/01_rosenblatt_forward.py')
    model = module.Perceptron()
    model.linear.weight.data[...] = [[2], [-1]]
    model.linear.bias.data[...] = 0.5
    inputs = Tensor([[0, 0], [1, 0], [0, 1]])
    expected = 1 / (1 + np.exp(-np.array([[0.5], [2.5], [-0.5]])))
    np.testing.assert_allclose(model(inputs).data, expected, rtol=1e-6)


def test_xor_crisis_shows_a_real_three_of_four_boundary():
    module = milestone('02_1969_xor/01_xor_crisis.py')
    model = module.SingleLayerPerceptron()
    dtype = model.linear.weight.data.dtype
    model.set_weights(1, 1, -0.5)
    accuracy, _ = module.evaluate_on_xor(model)
    assert accuracy == 0.75
    assert model.get_weights() == (1, 1, -0.5)
    assert model.linear.weight.data.dtype == dtype


def test_failed_xor_training_does_not_report_process_success(monkeypatch):
    module = milestone('02_1969_xor/02_xor_solved.py')
    monkeypatch.setattr(module, 'train_network', lambda *args, **kwargs:
                        {'loss': [1.0], 'accuracy': [0.5]})
    assert module.main() == 1
