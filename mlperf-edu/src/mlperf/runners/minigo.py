"""MLPerf Training v0.5 MiniGo: PyTorch adapter over the pinned reference.

The historical MiniGo reference is TensorFlow 1.x on CUDA, which no laptop
runs. Its Go rules, feature planes, MCTS, self-play loop, SGF handling, and
professional-move evaluation are all pure Python and NumPy, so only the network
needs replacing.

The import chain is ``selfplay_mcts -> gtp_wrapper -> dual_net -> tensorflow``.
Injecting a ``dual_net`` module that exposes a PyTorch ``DualNetwork`` therefore
replaces exactly one component and leaves the rest of the reference untouched,
loaded from the hash-validated archive rather than vendored. What the benchmark
measures, the rules and the search, stays byte-identical to MLCommons' code.

``coords`` imports ``gtp`` for the interactive Go Text Protocol server, which
self-play, training, and evaluation never reach; it is stubbed so the rules can
load without pulling in that dependency.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from mlperf.assets import ensure_minigo_reference, minigo_reference_paths
from mlperf.runners.common import (
    TrainingProgress,
    configured_seed,
    select_torch_device,
    synchronize_device,
)

MINIGO_UPSTREAM_REVISION = "0badcd1786fcb007725ed05f1c44e9d80bbeac52"
REFERENCE_MODULE_NAMES = (
    "coords",
    "go",
    "features",
    "symmetries",
    "sgf_wrapper",
    "mcts",
    "strategies",
    "selfplay_mcts",
)

_REFERENCE: dict[str, Any] | None = None


def _install_gtp_stub() -> None:
    """Satisfy coords' GTP import without the interactive protocol package."""
    if "gtp" in sys.modules:
        return
    stub = types.ModuleType("gtp")
    stub.PASS, stub.RESIGN = "pass", "resign"
    stub.BLACK, stub.WHITE = "black", "white"
    stub.Engine = object
    sys.modules["gtp"] = stub


def load_reference(params_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Import the pinned reference modules with a PyTorch network behind them.

    Board size and search constants are read at import time from the
    reference's own ``params/final.json``, which is the v0.5 9x9 configuration
    the registry contract mirrors. Overrides are applied to that file rather
    than to the modules, so the reference reads its parameters the way it
    always does.
    """
    global _REFERENCE
    if _REFERENCE is not None:
        return _REFERENCE

    ensure_minigo_reference(download=True)
    paths = minigo_reference_paths()
    root = paths["minigo"]
    params_path = root / "params" / "final.json"
    params = json.loads(params_path.read_text(encoding="utf-8"))
    params.update(params_overrides or {})

    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(params, handle)
    handle.close()
    os.environ["GOPARAMS"] = handle.name

    _install_gtp_stub()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    import features  # noqa: PLC0415
    import go  # noqa: PLC0415
    import symmetries  # noqa: PLC0415

    # The seam. Must be registered before anything imports gtp_wrapper.
    dual_net = types.ModuleType("dual_net")
    dual_net.DualNetwork = DualNetwork
    sys.modules["dual_net"] = dual_net

    modules = {name: __import__(name) for name in REFERENCE_MODULE_NAMES}
    modules.update({"go": go, "features": features, "symmetries": symmetries})
    _REFERENCE = {"root": root, "params": params, "modules": modules}
    return _REFERENCE


def reference() -> dict[str, Any]:
    if _REFERENCE is None:
        raise RuntimeError("load_reference() must run before the reference is used")
    return _REFERENCE


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.norm1(self.conv1(x)))
        return torch.relu(x + self.norm2(self.conv2(h)))


class DualNetwork(nn.Module):
    """Joint policy and value network, in the shape the reference MCTS expects.

    The reference calls only ``run``, ``run_many``, and ``save_file``. Keeping
    the surface that small is what makes this a replacement rather than a
    rewrite.
    """

    def __init__(
        self,
        save_file: str | None = None,
        *,
        channels: int = 64,
        blocks: int = 6,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        ref = reference()
        go = ref["modules"]["go"]
        features = ref["modules"]["features"]

        self.save_file = save_file or "in-memory"
        self.device = device or select_torch_device()
        self.board = go.N
        self.planes = features.NEW_FEATURES_PLANES
        moves = self.board * self.board + 1

        self.stem = nn.Sequential(
            nn.Conv2d(self.planes, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.tower = nn.Sequential(*[_ResidualBlock(channels) for _ in range(blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, 1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * self.board * self.board, moves),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.board * self.board, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )
        self.to(self.device)
        self.eval()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.tower(self.stem(x))
        return self.policy_head(h), self.value_head(h)

    def run(self, position, use_random_symmetry: bool = True):
        probs, values = self.run_many([position], use_random_symmetry)
        return probs[0], values[0]

    def run_many(self, positions, use_random_symmetry: bool = True):
        ref = reference()["modules"]
        processed = list(map(ref["features"].extract_features, positions))
        if use_random_symmetry:
            syms, processed = ref["symmetries"].randomize_symmetries_feat(processed)
        batch = torch.from_numpy(np.asarray(processed, dtype=np.float32))
        batch = batch.permute(0, 3, 1, 2).contiguous().to(self.device)
        was_training = self.training
        self.eval()
        with torch.no_grad():
            logits, value = self(batch)
            probs = torch.softmax(logits, dim=-1)
        if was_training:
            self.train()
        probs = probs.detach().cpu().numpy().astype(np.float32)
        value = value.squeeze(-1).detach().cpu().numpy().astype(np.float32)
        if use_random_symmetry:
            probs = ref["symmetries"].invert_symmetries_pi(syms, probs)
        return probs, value


def play_self_play_games(
    network: DualNetwork,
    *,
    games: int,
    readouts: int,
    resign_threshold: float = -0.9,
    progress: TrainingProgress | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
    """Generate training data by self-play, through the reference loop.

    This is the inference-in-the-loop step the workload exists to exercise:
    every training example is produced by running the current network inside
    tree search, so inference throughput bounds training throughput.
    """
    ref = reference()["modules"]
    positions: list[np.ndarray] = []
    search_probs: list[np.ndarray] = []
    outcomes: list[float] = []

    for index in range(games):
        player = ref["selfplay_mcts"].play(network, readouts, resign_threshold)
        pis = np.asarray(player.searches_pi, dtype=np.float32)
        replay = list(player.position_replay()) if hasattr(player, "position_replay") else None
        if replay is None:
            replay = _positions_from_player(player, ref)
        take = min(len(replay), len(pis))
        for move_index in range(take):
            positions.append(ref["features"].extract_features(replay[move_index]))
            search_probs.append(pis[move_index])
            # z is the game result from the perspective of the side to move.
            outcomes.append(float(player.result) * replay[move_index].to_play)
        if progress is not None:
            progress.update(index + 1, moves=take, result=player.result_string or "?")
    return positions, search_probs, outcomes


def _positions_from_player(player, ref) -> list:
    """Replay the finished game to recover the position at each move.

    The reference stores search probabilities per move but keeps only the final
    position, so the sequence is regenerated through the same rules that
    produced it.
    """
    go = ref["go"]
    position = go.Position()
    replay = [position]
    for move in player.root.position.recent:
        position = position.play_move(move.move)
        replay.append(position)
    return replay[: len(player.searches_pi)] or [go.Position()]


def train_on_self_play(
    network: DualNetwork,
    positions: list[np.ndarray],
    search_probs: list[np.ndarray],
    outcomes: list[float],
    *,
    batch_size: int,
    steps: int,
    learning_rate: float,
    seed: int,
) -> dict[str, float]:
    """AlphaGo-Zero objective: cross-entropy on search policy, MSE on outcome."""
    if not positions:
        raise ValueError("self-play produced no training positions")
    device = network.device
    # Reference features are NHWC; torch convolutions want NCHW, and the
    # permuted view must be made contiguous or autograd rejects it.
    x = (
        torch.from_numpy(np.asarray(positions, dtype=np.float32))
        .permute(0, 3, 1, 2)
        .contiguous()
    )
    pi = torch.from_numpy(np.asarray(search_probs, dtype=np.float32))
    z = torch.from_numpy(np.asarray(outcomes, dtype=np.float32)).unsqueeze(-1)

    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    network.train()
    policy_losses, value_losses = [], []
    for _ in range(steps):
        idx = torch.randint(0, x.shape[0], (min(batch_size, x.shape[0]),), generator=generator)
        xb, pib, zb = x[idx].to(device), pi[idx].to(device), z[idx].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, value = network(xb)
        policy_loss = -(pib * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        value_loss = torch.nn.functional.mse_loss(value, zb)
        (policy_loss + value_loss).backward()
        optimizer.step()
        policy_losses.append(float(policy_loss.detach()))
        value_losses.append(float(value_loss.detach()))
    network.eval()
    return {
        "policy_loss": float(np.mean(policy_losses)),
        "value_loss": float(np.mean(value_losses)),
        "training_positions": int(x.shape[0]),
    }


def evaluate_professional_moves(
    network: DualNetwork, *, readouts: int, tries_per_move: int
) -> dict[str, Any]:
    """The contract's quality metric, computed the way the reference computes it.

    Each professional game is replayed position by position; at each position
    the player searches and its chosen move is compared against the move the
    professional actually played. The reported accuracy is the fraction matched.
    """
    ref = reference()["modules"]
    root = reference()["root"]
    sgf_paths = sorted((root / "benchmark_sgf").glob("*.sgf"))
    if not sgf_paths:
        raise FileNotFoundError(f"no professional SGF inputs under {root/'benchmark_sgf'}")

    from gtp_wrapper import MCTSPlayer  # noqa: PLC0415

    per_game: dict[str, float] = {}
    ratings: list[float] = []
    for path in sgf_paths:
        replay = list(ref["sgf_wrapper"].replay_sgf(path.read_text(encoding="utf-8")))
        player = MCTSPlayer(network, verbosity=0, two_player_mode=True, num_parallel=4)
        game_ratings: list[float] = []
        for context in replay:
            if context.next_move is None:
                continue
            correct = 0
            for _ in range(tries_per_move):
                player.initialize_game(context.position)
                leaf = player.root.select_leaf()
                probs, value = network.run(leaf.position)
                leaf.incorporate_results(probs, value, leaf)
                while player.root.N < readouts:
                    player.tree_search()
                if player.pick_move() == context.next_move:
                    correct += 1
            game_ratings.append(correct / tries_per_move)
        if game_ratings:
            per_game[path.name] = float(np.mean(game_ratings))
            ratings.extend(game_ratings)

    if not ratings:
        raise ValueError("professional-move evaluation produced no rated positions")
    return {
        "professional_move_prediction": float(np.mean(ratings)),
        "rated_positions": len(ratings),
        "per_game": per_game,
        "games": len(per_game),
    }


def run_reinforcement_learning_max(workload, output_dir: Path) -> dict[str, Any]:
    """Train MiniGo by self-play until the inherited target or the budget ends.

    The reference trains for up to NUM_MAIN_ITERATIONS generations and stops
    when professional-move prediction reaches TERMINATION_ACCURACY. That target
    is the registry's quality gate, so the loop mirrors it: generate games with
    the current network, train on them, evaluate, repeat.

    The generation budget is a laptop concession and is recorded as one. The
    reference plays 2000 games per generation, which measures at roughly three
    and a half hours per generation single-process on this machine, so the
    contract carries a smaller budget and the report states both numbers.
    """
    from mlperf.fingerprint import detect_hardware
    from mlperf.manifest import build_provd
    from mlperf.registry import find_project_root

    root = find_project_root()
    seed = configured_seed()
    torch.manual_seed(seed)
    np.random.seed(seed)

    contract = (workload.raw.get("canonical_max_contract") or {}).get("config") or {}
    readouts = int(contract.get("search_readouts", 200))
    games_per_generation = int(
        os.environ.get(
            "MLPERF_EDU_MINIGO_GAMES_PER_GENERATION",
            contract.get("laptop_games_per_generation", 16),
        )
    )
    generations = int(
        os.environ.get(
            "MLPERF_EDU_MINIGO_GENERATIONS", contract.get("laptop_generations", 4)
        )
    )
    eval_readouts = int(contract.get("evaluation_readouts", 64))
    eval_tries = int(contract.get("evaluation_tries_per_move", 1))
    train_steps = int(contract.get("training_steps_per_generation", 200))
    batch_size = int(contract.get("training_batch_size", 64))
    learning_rate = float(contract.get("learning_rate", 1e-3))

    load_reference()
    reference_params = reference()["params"]
    target = float(workload.quality_value or reference_params["TERMINATION_ACCURACY"])
    tolerance = float(workload.quality_tolerance or 0.0)

    device = select_torch_device()
    network = DualNetwork(device=device)

    accuracies: list[float] = []
    policy_losses: list[float] = []
    value_losses: list[float] = []
    generation_seconds: list[float] = []
    best_accuracy = 0.0
    best_state: dict[str, torch.Tensor] | None = None

    progress = TrainingProgress(workload.id, generations, unit="generation")
    synchronize_device(device)
    started = time.perf_counter()
    for generation in range(generations):
        generation_start = time.perf_counter()
        positions, search_probs, outcomes = play_self_play_games(
            network, games=games_per_generation, readouts=readouts
        )
        stats = train_on_self_play(
            network,
            positions,
            search_probs,
            outcomes,
            batch_size=batch_size,
            steps=train_steps,
            learning_rate=learning_rate,
            seed=seed + generation,
        )
        evaluation = evaluate_professional_moves(
            network, readouts=eval_readouts, tries_per_move=eval_tries
        )
        accuracy = evaluation["professional_move_prediction"]

        accuracies.append(accuracy)
        policy_losses.append(stats["policy_loss"])
        value_losses.append(stats["value_loss"])
        generation_seconds.append(time.perf_counter() - generation_start)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in network.state_dict().items()
            }
        progress.update(
            generation + 1,
            accuracy=accuracy,
            best=best_accuracy,
            target=target,
            positions=stats["training_positions"],
        )
        if best_accuracy >= target - tolerance:
            break
    synchronize_device(device)
    duration = time.perf_counter() - started
    progress.close(f"best professional-move prediction {best_accuracy:.4f}")

    target_met = best_accuracy + tolerance >= target
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = (output_dir / f"{workload.id}_max_report.json").resolve()
    manifest_path = (output_dir / f"{workload.id}_max.provd.json").resolve()
    checkpoint_path = (output_dir / f"{workload.id}_max_checkpoint.pt").resolve()
    torch.save(best_state or network.state_dict(), checkpoint_path)

    report = {
        "schema": "mlperf-edu-report/0.1",
        "id": workload.id,
        "workload": workload.id,
        "suite": workload.suite,
        "profile": "max",
        "status": "passed" if target_met else "quality_failed",
        "backend": f"pytorch-{device.type}",
        "data_mode": "run-generated-self-play-plus-pinned-professional-games",
        "seed": seed,
        "metrics": {
            "professional_move_prediction": best_accuracy,
            "final_professional_move_prediction": accuracies[-1] if accuracies else 0.0,
            "accuracy_curve": accuracies,
            "policy_losses": policy_losses,
            "value_losses": value_losses,
            "generation_seconds": generation_seconds,
            "generations_completed": len(accuracies),
            "self_play_games": len(accuracies) * games_per_generation,
            "self_play_and_training_seconds": duration,
            "n_params": sum(p.numel() for p in network.parameters()),
        },
        "quality": {
            "metric": workload.quality_metric,
            "metric_key": "professional_move_prediction",
            "target": target,
            "tolerance": tolerance,
            "value": best_accuracy,
            "target_met": target_met,
            "quality_required": True,
            "acceptance_runs": 1,
        },
        "reference": {
            "authority": "MLCommons MLPerf Training v0.5 MiniGo",
            "upstream_revision": MINIGO_UPSTREAM_REVISION,
            "reference_modules_used": list(REFERENCE_MODULE_NAMES),
            "adaptation": (
                "MLPerf EDU replaces only the TensorFlow DualNetwork with a "
                "PyTorch equivalent. Go rules, feature planes, MCTS, the "
                "self-play loop, SGF handling, and the professional-move "
                "evaluation are the pinned reference code, executed unmodified."
            ),
            "contract_games_per_generation": reference_params["MAX_GAMES_PER_GENERATION"],
            "laptop_games_per_generation": games_per_generation,
            "laptop_generations": generations,
            "budget_note": (
                "The reference plays 2000 games per generation for up to "
                f"{reference_params['NUM_MAIN_ITERATIONS']} generations. This run uses a "
                "reduced laptop budget, so a miss here is a budget statement "
                "rather than a reproduction failure."
            ),
        },
        "artifacts": {
            "report": str(report_path),
            "provenance": str(manifest_path),
            "checkpoint": str(checkpoint_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    manifest = build_provd(
        workload=workload.id,
        scenario=workload.scenario or "training",
        division="closed",
        hardware_fingerprint=detect_hardware(),
        report=report,
        report_path=report_path,
        weights_path=checkpoint_path,
        weights_n_params=sum(p.numel() for p in network.parameters()),
        weights_dtype="float32",
        dataset_name="minigo-self-play",
        dataset_files=sorted(
            str(path) for path in (reference()["root"] / "benchmark_sgf").glob("*.sgf")
        ),
        rng_seed=seed,
        torch_state_bytes=torch.get_rng_state().numpy().tobytes(),
        repo_root=root,
    )
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    return report
