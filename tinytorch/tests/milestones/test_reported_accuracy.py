"""Final milestone metrics cannot be replaced by targets or progress bars."""
import pytest
from test_milestones_run import reported_accuracy


def test_accuracy_uses_named_result_not_higher_target_or_training_score():
    output = "Progress 100%\nTrain Accuracy │ 99.0%\nTest Accuracy │ 42.0%\nTarget 95%"
    assert reported_accuracy(output, "Test Accuracy") == 42.0
    assert reported_accuracy("│ 1. Reversal │ 45.0% │ 95% │ FAILED │", "1. Reversal") == 45.0


@pytest.mark.parametrize("output", ["Progress 100%", "Test Accuracy target 95%", "Test Accuracy │ 101%",
                                   "Test Accuracy │ 80%\nTest Accuracy │ 90%"])
def test_missing_invalid_or_ambiguous_metric_is_rejected(output):
    with pytest.raises(AssertionError):
        reported_accuracy(output, "Test Accuracy")


def test_terminal_colors_do_not_hide_the_result():
    assert reported_accuracy("Test Accuracy │ \x1b[32m81.5%\x1b[0m", "Test Accuracy") == 81.5
