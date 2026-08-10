"""The MiniGo adapter must replace the network and nothing else.

The value of this workload is that the Go rules, feature planes, MCTS, and
professional-move evaluation are MLCommons' code executing unmodified. If the
adapter ever grows past the network seam, that claim quietly stops being true,
so these tests pin the boundary rather than the behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlperf.runners import minigo


@pytest.fixture(scope="module")
def reference():
    return minigo.load_reference()


def test_reference_modules_load_from_the_pinned_archive(reference):
    root = reference["root"]
    assert minigo.MINIGO_UPSTREAM_REVISION in str(root)
    for name in minigo.REFERENCE_MODULE_NAMES:
        module = reference["modules"][name]
        assert str(root) in getattr(module, "__file__", ""), name


def test_board_and_search_come_from_the_reference_parameters(reference):
    params = reference["params"]
    go = reference["modules"]["go"]
    assert go.N == params["BOARD_SIZE"] == 9
    assert params["SP_READOUTS"] == 200
    assert params["MAX_GAMES_PER_GENERATION"] == 2000
    # The registry's quality target is the reference's own termination accuracy.
    assert params["TERMINATION_ACCURACY"] == 0.4


def test_network_satisfies_only_the_interface_mcts_uses(reference):
    network = minigo.DualNetwork()
    for attribute in ("run", "run_many", "save_file"):
        assert hasattr(network, attribute), attribute

    go = reference["modules"]["go"]
    position = go.Position()
    probs, value = network.run(position)
    moves = go.N * go.N + 1
    assert probs.shape == (moves,)
    assert np.isfinite(probs).all()
    assert abs(float(probs.sum()) - 1.0) < 1e-3, "policy head must be a distribution"
    assert -1.0 <= float(value) <= 1.0, "value head is tanh-bounded"


def test_professional_evaluation_reads_all_four_pinned_games(reference):
    sgf = sorted((reference["root"] / "benchmark_sgf").glob("*.sgf"))
    assert len(sgf) == 4, [path.name for path in sgf]


def test_untrained_network_scores_near_chance():
    """A correctness check on the metric, not on the model.

    An untrained policy should predict a professional's move about as often as
    picking uniformly among legal points. A number far from chance would mean
    the evaluation is scoring something other than move agreement.
    """
    ref = minigo.load_reference()
    network = minigo.DualNetwork()
    result = minigo.evaluate_professional_moves(network, readouts=8, tries_per_move=1)

    go = ref["modules"]["go"]
    chance = 1.0 / (go.N * go.N + 1)
    assert result["games"] == 4
    assert result["rated_positions"] > 100
    assert 0.0 <= result["professional_move_prediction"] < 12 * chance
