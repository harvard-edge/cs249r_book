"""Distillation must be a trainable, batch-mean objective (2026-09-11)."""
import numpy as np
import pytest
from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear
from tinytorch.core.optimizers import SGD
from tinytorch.perf.compression import KnowledgeDistillation


def test_distillation_gradient_matches_analytic_mean_objective():
    student_data = np.array([[1., 0., -1.], [0., 2., 1.]], dtype=np.float32)
    teacher_data = np.array([[0., 2., 1.], [1., 0., -1.]], dtype=np.float32)
    labels = np.array([1, 0])
    student = Tensor(student_data, requires_grad=True)
    teacher = Tensor(teacher_data, requires_grad=True)
    temperature, alpha = 2.0, 0.7
    loss = KnowledgeDistillation(None, None, temperature, alpha).distillation_loss(student, teacher, Tensor(labels))
    assert isinstance(loss, Tensor) and loss.shape == ()
    loss.backward()
    def softmax(x):
        e = np.exp(x - x.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
    expected = alpha * (softmax(student_data / temperature) - softmax(teacher_data / temperature)) / temperature
    expected += (1 - alpha) * (softmax(student_data) - np.eye(3)[labels])
    np.testing.assert_allclose(student.grad, expected / 2, rtol=1e-5, atol=1e-7)
    assert teacher.grad is None


def test_label_representations_and_batch_duplication_agree():
    kd = KnowledgeDistillation(None, None)
    student, teacher = np.array([[1., 0.], [0., 2.]]), np.array([[0., 1.], [2., 0.]])
    labels = np.array([0, 1])
    expected = kd.distillation_loss(Tensor(student), Tensor(teacher), labels).data
    for targets in [Tensor(labels), np.eye(2)[labels], Tensor(np.eye(2)[labels])]:
        np.testing.assert_allclose(kd.distillation_loss(Tensor(student), Tensor(teacher), targets).data, expected)
    duplicated = kd.distillation_loss(Tensor(np.tile(student, (2, 1))), Tensor(np.tile(teacher, (2, 1))), np.tile(labels, 2))
    np.testing.assert_allclose(duplicated.data, expected)


def test_optimizer_reduces_distillation_loss_without_changing_teacher():
    teacher, student = Linear(2, 2), Linear(2, 2)
    teacher.weight.data[:] = [[2., -2.], [-2., 2.]]
    teacher.bias.data[:] = 0
    student.weight.data[:] = 0
    student.bias.data[:] = 0
    before = [p.data.copy() for p in teacher.parameters()]
    x, labels = Tensor([[1., 0.], [0., 1.]]), Tensor([0, 1])
    targets = teacher(x)
    kd = KnowledgeDistillation(teacher, student)
    optimizer = SGD(student.parameters(), lr=0.2)
    initial = float(kd.distillation_loss(student(x), targets, labels).data)
    for _ in range(20):
        optimizer.zero_grad()
        loss = kd.distillation_loss(student(x), targets, labels)
        loss.backward()
        optimizer.step()
    assert float(kd.distillation_loss(student(x), targets, labels).data) < initial * 0.8
    for param, original in zip(teacher.parameters(), before):
        np.testing.assert_array_equal(param.data, original)
        assert param.grad is None


@pytest.mark.parametrize("labels", [[0.5], [-1], [2], [np.nan], [[0.2, 0.2]], [[-1., 2.]]])
def test_invalid_labels_are_rejected(labels):
    with pytest.raises(ValueError):
        KnowledgeDistillation(None, None).distillation_loss(Tensor([[1., 0.]]), Tensor([[0., 1.]]), Tensor(labels))


def test_extreme_logits_have_finite_loss_and_gradients():
    logits = Tensor([[1000., -1000.]], requires_grad=True)
    loss = KnowledgeDistillation(None, None).distillation_loss(logits, Tensor([[-1000., 1000.]]), Tensor([1]))
    loss.backward()
    assert np.isfinite(loss.data) and np.all(np.isfinite(logits.grad))
