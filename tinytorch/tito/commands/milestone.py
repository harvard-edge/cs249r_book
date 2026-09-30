"""
Milestone command group for TinyTorch CLI: capability-based learning progression.

The milestone system transforms module completion into meaningful capability achievements.
Instead of just finishing modules, students unlock epic milestones that represent
real-world ML engineering skills.
"""

from argparse import ArgumentParser, Namespace
from rich.panel import Panel
from rich import box
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn, TimeElapsedColumn
from rich.console import Console
from rich.align import Align
from rich.text import Text
from rich.layout import Layout
from rich.tree import Tree
from rich.columns import Columns
from rich.cells import cell_len
import sys
import os
import json
import time
import signal
import subprocess
import importlib
import yaml
from datetime import datetime
from pathlib import Path

from .base import BaseCommand
from ..core.console import print_ascii_logo
from ..core.console import get_console
from ..core import milestone_tracker


# Name aliases for milestone IDs (allows `tito milestone run perceptron`)
MILESTONE_ALIASES = {
    "perceptron": "01",
    "xor": "02",
    "mlp": "03",
    "cnn": "04",
    "transformer": "05",
    "tinygpt": "05",
    "gpt": "05",
    "shakespeare": "05",
    "code": "05",
    "tinypy": "05",
    "tinycopilot": "05",
    "copilot": "05",
    "mlperf": "06",
    "olympics": "06",
    "serving": "06",
    "kernels": "07",
    "triton": "07",
    "metal": "07",
    "extensions": "07",
    "accelerators": "07",
}

# Milestone-to-script mapping for tito milestone run command
#
# required_modules rule (audited 2026-09 against cProfile traces of each part):
# list every module whose code the part actually runs, plus each of those
# modules' own declared prerequisites (the "**Prerequisites**" line in its
# Module Dependencies section). A module that neither runs nor is such a
# prerequisite is not listed, so it cannot gate the unlock. The top-level list
# is the union of the per-part lists.
MILESTONE_SCRIPTS = {
    "01": {
        "id": "01",
        "name": "Perceptron (1958)",
        "year": 1958,
        "title": "Frank Rosenblatt's First Neural Network",
        "script": "milestones/01_1958_perceptron/01_rosenblatt_forward.py",
        "required_parts": [1],
        "required_modules": [1, 2, 3],  # Tensor, Activations, Layers (forward pass only)
        "description": "Build the first neural network (forward pass)",
        "historical_context": "Rosenblatt's perceptron proved machines could learn",
        "emoji": "🧠"
    },
    "02": {
        "id": "02",
        "name": "XOR Crisis (1969)",
        "year": 1969,
        "title": "The Problem That Stalled AI",
        "script": "milestones/02_1969_xor/01_xor_crisis.py",
        "required_parts": [1],
        "required_modules": [1, 2, 3],  # Just forward pass: Tensor, Activations, Layers
        "description": "Single-layer perceptron CANNOT solve XOR (75% max)",
        "historical_context": "Minsky & Papert proved limits of single-layer networks",
        "emoji": "🔀"
    },
    "03": {
        "id": "03",
        "name": "MLP Revival (1986)",
        "year": 1986,
        "title": "Backpropagation Breakthrough",
        # Both parts are the milestone: XOR shows hidden layers are sufficient,
        # TinyDigits shows the same MLP learns real data.
        "required_parts": [1, 2],
        "scripts": [
            {
                "name": "XOR Solved",
                "script": "milestones/02_1969_xor/02_xor_solved.py",
                "description": "Hidden layers + backprop SOLVE the impossible XOR problem!",
                # Runs 01-04, 06, 07 (manual loop, no DataLoader or Trainer).
                "required_modules": [1, 2, 3, 4, 6, 7]
            },
            {
                "name": "TinyDigits",
                "script": "milestones/03_1986_mlp/01_rumelhart_tinydigits.py",
                "description": "Scale up to real data - handwritten digit recognition",
                # Runs 01-07 (YOUR DataLoader, manual loop; no Trainer).
                "required_modules": [1, 2, 3, 4, 5, 6, 7]
            }
        ],
        "required_modules": [1, 2, 3, 4, 5, 6, 7],  # Union of both parts
        "description": "Solve XOR with hidden layers, then train on real data",
        "historical_context": "Rumelhart, Hinton & Williams (Nature, 1986) ended the AI Winter",
        "emoji": "🎓"
    },
    "04": {
        "id": "04",
        "name": "CNN Revolution (1998)",
        "year": 1998,
        "title": "LeNet - Computer Vision Breakthrough",
        # Part 1 (TinyDigits, offline) is the milestone. Part 2 (CIFAR-10) is an
        # optional extension: it needs a large download, so it is recorded per
        # part when run but never required for completion.
        "required_parts": [1],
        "scripts": [
            {
                "name": "TinyDigits",
                "script": "milestones/04_1998_cnn/01_lecun_tinydigits.py",
                "description": "Train a LeNet-style CNN on 8x8 handwritten digits (works offline)",
                # Runs 01-07 and 09 (manual loop; no Trainer).
                "required_modules": [1, 2, 3, 4, 5, 6, 7, 9]
            },
            {
                "name": "CIFAR-10",
                "script": "milestones/04_1998_cnn/02_lecun_cifar10.py",
                "description": "Scale to natural images with YOUR DataLoader (requires download)",
                # Imports 01-05, 07, 09 (+06 via autograd); no Trainer.
                "required_modules": [1, 2, 3, 4, 5, 6, 7, 9]
            }
        ],
        "required_modules": [1, 2, 3, 4, 5, 6, 7, 9],  # Union of both parts
        "description": "Build LeNet for digit recognition, then scale to natural images",
        "historical_context": "Yann LeCun's convolutional networks revolutionized computer vision",
        "emoji": "👁️"
    },
    "05": {
        "id": "05",
        "name": "Transformer Era (2017)",
        "year": 2017,
        "title": "TinyGPT: Autoregressive Language Modeling (ChatGPT Foundation)",
        # Parts 1-2 are the milestone: train an autoregressive transformer
        # (TinyGPT) and prove attention routes information (sequence tasks).
        # Parts 3-4 (TinyCopilot, Conversational Q&A) are optional extensions:
        # they reuse the same modules on new corpora and teach no new mechanism,
        # so they are recorded per part when run but never required.
        "required_parts": [1, 2],
        "scripts": [
            {
                "name": "TinyGPT (Shakespeare)",
                "script": "milestones/05_2017_transformer/01_tinygpt_shakespeare.py",
                "description": "Train TinyGPT from scratch on Shakespeare and generate text",
                "required_modules": [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13]
            },
            {
                "name": "Sequence Routing",
                "script": "milestones/05_2017_transformer/02_vaswani_attention.py",
                "description": "Prove attention mechanism on sequence reversal and copying",
                # Runs 01-07 and 11-13 (no Trainer, no tokenizer).
                "required_modules": [1, 2, 3, 4, 5, 6, 7, 11, 12, 13]
            },
            {
                "name": "TinyCopilot",
                "script": "milestones/05_2017_transformer/03_tinycopilot.py",
                "description": "Train TinyCopilot on Python code to complete functions and syntax",
                "required_modules": [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13]
            },
            {
                "name": "Conversational Q&A",
                "script": "milestones/05_2017_transformer/04_tinygpt_chat.py",
                "description": "Train conversational TinyGPT on TinyTorch Q&A and analyze overfitting",
                # Measured 2026-09 as running 01-08 and 11-13 only: Module 10 is
                # listed because this part is being switched to the student's
                # Module 10 tokenizer. Drop 10 if that switch does not land.
                "required_modules": [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13]
            }
        ],
        "required_modules": [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13],  # TinyGPT training requirements
        "description": "Train TinyGPT from scratch on Shakespeare, TinyCopilot code generation, and concept Q&A",
        "historical_context": "Vaswani et al. (2017) and the Generative LLM revolution (2020-2022) proved transformers and emergent autoregressive scaling",
        "emoji": "🤖"
    },
    "06": {
        "id": "06",
        "name": "MLPerf to Generative Serving (2018)",
        "year": 2018,
        "title": "MLPerf to ChatGPT Serving (The Optimization Olympics)",
        # Both parts are the milestone: compression (Part 1) and generation
        # speedup via KV caching (Part 2) cover different optimization modules.
        "required_parts": [1, 2],
        "scripts": [
            {
                "name": "Model Compression",
                "script": "milestones/06_2018_mlperf/01_optimization_olympics.py",
                "description": "Profiling + Quantization + Pruning on MLP",
                # Runs 01-04, 06, 07, 09, 11-19 (the GPT candidates need 11-13;
                # Module 17 kernels need 09). No DataLoader or Trainer.
                "required_modules": [1, 2, 3, 4, 6, 7, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19]
            },
            {
                "name": "Generation Speedup",
                "script": "milestones/06_2018_mlperf/02_generation_speedup.py",
                "description": "Verify cached GPT outputs and measure inference speed",
                # Runs 01-03, 06, 11-13, 18; plus 04 (prerequisite of 06) and
                # 14 (prerequisite of 18).
                "required_modules": [1, 2, 3, 4, 6, 11, 12, 13, 14, 18]
            }
        ],
        "required_modules": [1, 2, 3, 4, 6, 7, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19],  # Union of both parts
        "description": "Compress and accelerate TinyGPT across the MLPerf Pareto frontier",
        "historical_context": "MLPerf standardized ML benchmarks (2018), paving the way for ChatGPT-scale production serving (2022)",
        "emoji": "🏆"
    },
    "07": {
        "id": "07",
        "name": "Custom Kernels (2024)",
        "year": 2024,
        "title": "Your Kernels vs. Native Silicon: C++ SIMD, Apple Metal & Triton",
        "script": "milestones/07_2024_kernels/01_custom_kernels.py",
        "required_parts": [1],
        # Only Module 17's code runs; 01, 06, 09, 14 are its declared
        # prerequisites (Module Dependencies), which its export depends on.
        "required_modules": [1, 6, 9, 14, 17],
        "description": "Verify YOUR Module 17 kernels on ragged shapes, then time them against bundled C++ SIMD, Metal, and Triton kernels",
        "historical_context": "Modern AI systems rely on custom GPU shaders to bypass framework overhead (2024)",
        "emoji": "⚡"
    }
}

# "What makes this special" bullets for the achievement panel, tailored to
# what each milestone's required_modules actually cover. Previously every
# milestone showed the same 3 lines including "Every gradient: YOUR
# autograd", which was wrong for 01 and 02 (forward-pass only, no autograd
# module required at all).
MILESTONE_ACHIEVEMENT_HIGHLIGHTS = {
    "01": [
        "Every line of code: YOUR implementations",
        "Every tensor operation: YOUR Tensor class",
        "Every forward pass: YOUR Layers (no autograd needed yet)",
    ],
    "02": [
        "Every line of code: YOUR implementations",
        "Every tensor operation: YOUR Tensor class",
        "The exact limitation Minsky & Papert proved in 1969",
    ],
    "03": [
        "Every line of code: YOUR implementations",
        "Every tensor operation: YOUR Tensor class",
        "Every gradient: YOUR autograd",
    ],
    "04": [
        "Every line of code: YOUR implementations",
        "Every convolution forward: YOUR Conv2d",
        "Every training step: YOUR optimizer and loss",
    ],
    "05": [
        "Every line of code: YOUR implementations",
        "Every attention score: YOUR Causal MultiHeadAttention",
        "Every token generated: YOUR autoregressive loop (ChatGPT foundation)",
    ],
    "06": [
        "Every line of code: YOUR implementations",
        "Every candidate measured: YOUR quantization and compression",
        "Every gate: YOUR quantizer, pruner, and KV cache checked against measured results",
    ],
    "07": [
        "Every line of code: YOUR implementations",
        "Every tiled matmul: YOUR blocked loop order, ragged tiles included",
        "Every convolution lowering: YOUR im2col and col2im",
        "Native C++/Metal/Triton kernels: bundled reference points, timed against yours",
    ],
}


MODULE_EXPORT_CHECKS = {
    1: [("tinytorch", "Tensor"), ("tinytorch.core.tensor", "Tensor")],
    2: [("tinytorch", "ReLU"), ("tinytorch.core.activations", "ReLU")],
    3: [("tinytorch", "Linear"), ("tinytorch.core.layers", "Linear")],
    4: [("tinytorch", "CrossEntropyLoss"), ("tinytorch.core.losses", "CrossEntropyLoss")],
    5: [("tinytorch", "DataLoader"), ("tinytorch.core.dataloader", "DataLoader")],
    6: [("tinytorch", "no_grad"), ("tinytorch.core.autograd", "no_grad")],
    7: [("tinytorch", "SGD"), ("tinytorch.core.optimizers", "SGD")],
    8: [("tinytorch", "Trainer"), ("tinytorch.core.training", "Trainer")],
    9: [("tinytorch", "Conv2d"), ("tinytorch.core.spatial", "Conv2d")],
    10: [("tinytorch", "CharTokenizer"), ("tinytorch.core.tokenization", "CharTokenizer")],
    11: [("tinytorch", "Embedding"), ("tinytorch.core.embeddings", "Embedding")],
    12: [("tinytorch", "MultiHeadAttention"), ("tinytorch.core.attention", "MultiHeadAttention")],
    13: [("tinytorch", "TransformerBlock"), ("tinytorch.core.transformers", "TransformerBlock")],
    14: [("tinytorch", "Profiler"), ("tinytorch.perf.profiling", "Profiler")],
    15: [("tinytorch", "Quantizer"), ("tinytorch.perf.quantization", "Quantizer")],
    16: [("tinytorch", "Compressor"), ("tinytorch.perf.compression", "Compressor")],
    17: [("tinytorch", "vectorized_matmul"), ("tinytorch.perf.acceleration", "vectorized_matmul")],
    18: [("tinytorch", "KVCache"), ("tinytorch.perf.memoization", "KVCache")],
    19: [("tinytorch.perf.benchmarking", "Benchmark")],
    20: [("tinytorch", "olympics")],
}


def _tito_dir() -> Path:
    """Student state lives in .tito/ under the directory tito runs from."""
    return Path(".tito")


def _module_progress_to_int(module_value):
    """Normalize module progress entries like 1, "01", or "01_tensor" to int."""
    if isinstance(module_value, int):
        return module_value
    if not isinstance(module_value, str):
        return None
    prefix = module_value.split("_", 1)[0]
    try:
        return int(prefix)
    except ValueError:
        return None


def _load_completed_module_numbers() -> set:
    """Read completed module numbers from the canonical .tito progress file."""
    progress_file = Path(".tito") / "progress.json"
    completed = set()
    if not progress_file.exists():
        return completed

    try:
        with open(progress_file, 'r', encoding='utf-8') as f:
            progress_data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return completed

    for module_value in progress_data.get("completed_modules", []):
        module_num = _module_progress_to_int(module_value)
        if module_num is not None:
            completed.add(module_num)
    return completed


def _required_modules_for(milestone: dict) -> list[int]:
    """Return all modules required by a milestone as sorted ints."""
    required = set()
    for module_value in milestone.get("required_modules", []):
        module_num = _module_progress_to_int(module_value)
        if module_num is not None:
            required.add(module_num)
    for script in milestone.get("scripts", []):
        for module_value in script.get("required_modules", []):
            module_num = _module_progress_to_int(module_value)
            if module_num is not None:
                required.add(module_num)
    return sorted(required)


def _validate_required_exports(required_modules: list[int]) -> list[str]:
    """Return missing or silently failed exports for the required modules."""
    failures = []

    for module_num in required_modules:
        for module_path, symbol_name in MODULE_EXPORT_CHECKS.get(module_num, []):
            try:
                module = importlib.import_module(module_path)
            except ImportError as exc:
                failures.append(f"{module_path}.{symbol_name}: import failed ({exc})")
                continue

            value = getattr(module, symbol_name, None)
            if value is None:
                failures.append(f"{module_path}.{symbol_name}: exported as None")

    return failures


def _warn_on_stale_exports(console, project_root, required_modules: list[int]) -> dict:
    """Warn (never fail) when a required module's notebook changed after export.

    The export check above only proves each symbol exists; it cannot tell
    whether the package holds the student's latest notebook code.
    """
    from .module.workflow import stale_export_report

    try:
        report = stale_export_report(Path(project_root), required_modules)
    except Exception:
        return {"stale": [], "unrecorded": []}
    for num in report["stale"]:
        console.print(
            f"  [bold yellow]⚠ Module {num} changed since you last exported; run "
            f"`tito module complete {num}` (or export) so the milestone runs your latest code[/bold yellow]"
        )
    if report["unrecorded"]:
        nums = ", ".join(report["unrecorded"])
        console.print(
            f"  [dim]• No export record for module(s) {nums}, so tito can't tell whether the "
            f"package matches your notebook. Re-run `tito module complete NN` to record it.[/dim]"
        )
    return report



# Exit codes of a milestone script stopped by Ctrl-C: killed by SIGINT (POSIX),
# the shell convention 128 + SIGINT, and STATUS_CONTROL_C_EXIT (Windows).
INTERRUPTED_EXIT_CODES = (-signal.SIGINT, 130, 0xC000013A)


def run_milestone_script(script_file) -> int:
    """Run one milestone script and return its exit code.

    Ctrl-C belongs to the script while it runs. The script decides whether it
    was interrupted mid-run (it then dies of SIGINT, see INTERRUPTED_EXIT_CODES)
    or the student left the post-pass try-it prompt (it exits 0 and the pass
    records). tito used to catch the same Ctrl-C itself and drop an earned
    pass (2026-09-29). A Python handler, unlike SIG_IGN, is not inherited by
    the child process, so the script still receives the default behavior.
    """
    previous_handler = signal.signal(signal.SIGINT, lambda signum, frame: None)
    try:
        return subprocess.run(
            [sys.executable, str(script_file)],
            capture_output=False,
            text=True, encoding="utf-8", errors="replace",
        ).returncode
    finally:
        signal.signal(signal.SIGINT, previous_handler)


class MilestoneSystem:
    """Core milestone tracking and management system."""

    def __init__(self, config):
        self.config = config
        self.console = get_console()

        # Load milestones from configuration file
        self.MILESTONES = self._load_milestones_config()

    def _load_milestones_config(self) -> dict:
        """Load milestone configuration from YAML files (main and era-specific)."""
        config_path = Path("milestones") / "milestones.yml"
        milestones = {}

        # Try to load main milestones.yml first
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)

                # Convert to expected format
                for milestone_id, milestone_data in config['milestones'].items():
                    milestone_data['id'] = str(milestone_id)
                    milestones[str(milestone_id)] = milestone_data

            except Exception as e:
                self.console.print(f"[yellow]Warning: Could not load main milestone config: {e}[/yellow]")

        # Also try to load era-specific configurations
        era_paths = [
            Path("milestones") / "foundation" / "milestone.yml",
            Path("milestones") / "revolution" / "milestone.yml",
            Path("milestones") / "generation" / "milestone.yml"
        ]

        for era_path in era_paths:
            if era_path.exists():
                try:
                    with open(era_path, 'r', encoding='utf-8') as f:
                        era_config = yaml.safe_load(f)

                    if 'milestone' in era_config:
                        milestone_data = era_config['milestone']
                        milestone_id = milestone_data['id']
                        milestones[str(milestone_id)] = milestone_data

                except Exception as e:
                    self.console.print(f"[yellow]Warning: Could not load era config {era_path}: {e}[/yellow]")

        # If no milestones loaded, use MILESTONE_SCRIPTS as fallback
        if not milestones:
            return MILESTONE_SCRIPTS

        return milestones

    def get_milestone_status(self) -> dict:
        """Get current milestone progress status."""
        milestone_data = self._get_milestone_progress_data()

        status = {
            "milestones": {},
            "overall_progress": 0,
            "total_unlocked": 0,
            "total_completed": 0,
            "next_milestone": None
        }

        total_milestones = len(self.MILESTONES)
        unlocked_count = 0
        completed_count = 0

        for milestone_id, milestone in self.MILESTONES.items():
            # Check if all required modules are complete (no more checkpoint dependencies)
            required_modules = _required_modules_for(milestone)
            required_complete = all(
                self._is_module_completed(f"{mod:02d}")
                for mod in required_modules
            )

            # Check if milestone is unlocked (ready to run, not the same as actually
            # run and achieved; see is_completed below)
            is_unlocked = milestone_id in milestone_data.get("unlocked_milestones", [])

            # Check if the milestone has actually been run and passed
            is_completed = milestone_id in milestone_data.get("completed_milestones", [])

            # Check if trigger module is completed (if trigger_module exists)
            trigger_module = milestone.get("trigger_module", "")
            if trigger_module:
                trigger_complete = self._is_module_completed(trigger_module)
            else:
                # No trigger module - consider complete if all required modules done
                trigger_complete = required_complete

            milestone_status = {
                "id": milestone_id,
                "name": milestone["name"],
                "title": milestone["title"],
                "emoji": milestone.get("emoji", "🎯"),
                "trigger_module": trigger_module,
                "required_modules": required_modules,
                "victory_condition": milestone.get("victory_condition", milestone.get("description", "")),
                "capability": milestone.get("capability", milestone.get("description", "")),
                "real_world_impact": milestone.get("real_world_impact", milestone.get("historical_context", "")),
                "required_complete": required_complete,
                "trigger_complete": trigger_complete,
                "is_unlocked": is_unlocked,
                "is_completed": is_completed,
                "can_unlock": required_complete and trigger_complete and not is_unlocked,
                "unlock_date": milestone_data.get("unlock_dates", {}).get(milestone_id)
            }

            status["milestones"][milestone_id] = milestone_status

            if is_completed:
                completed_count += 1
            if is_unlocked:
                unlocked_count += 1
            elif milestone_status["can_unlock"] and not status["next_milestone"]:
                status["next_milestone"] = milestone_id

        status["total_unlocked"] = unlocked_count
        status["total_completed"] = completed_count
        status["overall_progress"] = (unlocked_count / total_milestones) * 100 if total_milestones > 0 else 0

        return status

    def check_milestone_unlock(self, completed_module: str) -> dict:
        """Check if completing a module unlocks a milestone."""
        result = {
            "milestone_unlocked": False,
            "milestone_id": None,
            "milestone_data": None,
            "celebration_needed": False
        }

        completed_num = _module_progress_to_int(completed_module)

        # Find milestones made runnable by this completed module.
        for milestone_id, milestone in self.MILESTONES.items():
            trigger_module = milestone.get("trigger_module")
            if trigger_module:
                should_check = trigger_module == completed_module
            else:
                required_modules = _required_modules_for(milestone)
                should_check = bool(required_modules) and completed_num == max(required_modules)

            if should_check:
                status = self.get_milestone_status()
                milestone_status = status["milestones"][milestone_id]

                if milestone_status["can_unlock"]:
                    # Unlock the milestone!
                    self._unlock_milestone(milestone_id)
                    result.update({
                        "milestone_unlocked": True,
                        "milestone_id": milestone_id,
                        "milestone_data": milestone,
                        "celebration_needed": True
                    })
                break

        return result

    def run_milestone_test(self, milestone_id: str) -> dict:
        """Run tests to validate milestone achievement."""
        if milestone_id not in self.MILESTONES:
            return {"success": False, "error": f"Milestone {milestone_id} not found"}

        milestone = self.MILESTONES[milestone_id]

        # Check all required modules are complete
        required_modules = milestone.get("required_modules", [])
        failed_modules = []

        for mod in required_modules:
            if not self._is_module_completed(f"{mod:02d}"):
                failed_modules.append(f"{mod:02d}")

        if failed_modules:
            return {
                "success": False,
                "error": f"Required modules not completed: {', '.join(failed_modules)}",
                "milestone_name": milestone["name"]
            }

        # Check trigger module completion
        trigger_module = milestone.get("trigger_module", "")
        if trigger_module and not self._is_module_completed(trigger_module):
            return {
                "success": False,
                "error": f"Trigger module {trigger_module} not completed",
                "milestone_name": milestone["name"]
            }

        # All tests passed
        return {
            "success": True,
            "milestone_id": milestone_id,
            "milestone_name": milestone["name"],
            "title": milestone.get("title", ""),
            "capability": milestone.get("capability", milestone.get("description", "")),
            "victory_condition": milestone.get("victory_condition", "")
        }

    def _unlock_milestone(self, milestone_id: str) -> None:
        """Record milestone unlock in progress tracking."""
        milestone_data = self._get_milestone_progress_data()

        if milestone_id not in milestone_data["unlocked_milestones"]:
            milestone_data["unlocked_milestones"].append(milestone_id)
            milestone_data["unlock_dates"][milestone_id] = datetime.now().isoformat()
            milestone_data["total_unlocked"] = len(milestone_data["unlocked_milestones"])

        self._save_milestone_progress_data(milestone_data)

    def _is_module_completed(self, module_name: str) -> bool:
        """Check if a module has been completed."""
        # Check module progress file
        progress_file = Path(".tito") / "progress.json"
        if progress_file.exists():
            try:
                with open(progress_file, 'r', encoding='utf-8') as f:
                    progress_data = json.load(f)
                    module_num = _module_progress_to_int(module_name)
                    completed_nums = {
                        _module_progress_to_int(mod)
                        for mod in progress_data.get("completed_modules", [])
                    }
                    return module_num in completed_nums
            except (json.JSONDecodeError, IOError):
                pass
        return False

    def _get_milestone_progress_data(self) -> dict:
        """Read the milestone ledger (normalized; see tito/core/milestone_tracker.py)."""
        return milestone_tracker.load(_tito_dir())

    def _save_milestone_progress_data(self, milestone_data: dict) -> None:
        milestone_tracker.save(_tito_dir(), milestone_data)


class MilestoneCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "milestone"

    @property
    def description(self) -> str:
        return "Milestone achievement and capability unlock commands"

    def add_arguments(self, parser: ArgumentParser) -> None:
        subparsers = parser.add_subparsers(
            dest='milestone_command',
            help='Milestone subcommands',
            metavar='SUBCOMMAND'
        )

        # List subcommand (NEW)
        list_parser = subparsers.add_parser(
            'list',
            help='List available milestones and their status'
        )
        list_parser.add_argument(
            '--simple',
            action='store_true',
            help='Show simple list (less detail)'
        )

        # Run subcommand (NEW)
        run_parser = subparsers.add_parser(
            'run',
            help='Run a milestone with prerequisite checking'
        )
        run_parser.add_argument(
            'milestone_id',
            help='Milestone ID (01-07) or name (perceptron, xor, mlp, cnn, transformer, mlperf, tinygpt, kernels)'
        )
        run_parser.add_argument(
            '--part',
            type=int,
            help='Run and record only this part (the milestone completes once every required part has passed)'
        )
        run_parser.add_argument(
            '--all',
            action='store_true',
            help='Run every part, including optional extensions'
        )
        run_parser.add_argument(
            '--skip-checks',
            action='store_true',
            help='Skip prerequisite checks and run as a demo; nothing is recorded'
        )
        run_parser.add_argument(
            '--non-interactive', '-y', '--yes',
            dest='non_interactive',
            action='store_true',
            help='Run non-interactively without prompting'
        )

        # Info subcommand (NEW)
        info_parser = subparsers.add_parser(
            'info',
            help='Show detailed information about a milestone'
        )
        info_parser.add_argument(
            'milestone_id',
            help='Milestone ID (01-07) or name (perceptron, xor, mlp, cnn, transformer, mlperf, tinygpt, kernels)'
        )

        # Status subcommand
        status_parser = subparsers.add_parser(
            'status',
            help='View milestone progress and achievements'
        )
        status_parser.add_argument(
            '--detailed',
            action='store_true',
            help='Show detailed milestone information'
        )

        # Timeline subcommand
        timeline_parser = subparsers.add_parser(
            'timeline',
            help='View milestone timeline and progression'
        )
        timeline_parser.add_argument(
            '--horizontal',
            action='store_true',
            help='Show horizontal progress bar instead of tree'
        )

        # Test subcommand
        test_parser = subparsers.add_parser(
            'test',
            help='Test milestone achievement requirements'
        )
        test_parser.add_argument(
            'milestone_id',
            nargs='?',
            help='Milestone ID to test (1-6), or test next available'
        )

        # Demo subcommand
        demo_parser = subparsers.add_parser(
            'demo',
            help='Run milestone capability demonstration'
        )
        demo_parser.add_argument(
            'milestone_id',
            help='Milestone ID to demonstrate (1-6)'
        )

    def run(self, args: Namespace) -> int:
        console = self.console

        if not hasattr(args, 'milestone_command') or not args.milestone_command:
            console.print(Panel(
                "[bold cyan]Milestone Commands[/bold cyan]\n\n"
                "Recreate ML history and achieve epic capabilities!\n\n"
                "Available subcommands:\n"
                "  • [bold]list[/bold]       - List available milestones\n"
                "  • [bold]run[/bold]        - Run a milestone (with prereq checks)\n"
                "  • [bold]info[/bold]       - Show detailed milestone information\n"
                "  • [bold]status[/bold]     - View progress and achievements\n"
                "  • [bold]timeline[/bold]   - View milestone timeline\n"
                "  • [bold]test[/bold]       - Test milestone requirements\n"
                "  • [bold]demo[/bold]       - Run capability demonstration\n\n"
                "[dim]Examples:[/dim]\n"
                "[dim]  tito milestone list[/dim]\n"
                "[dim]  tito milestone run 03           # Run every required part[/dim]\n"
                "[dim]  tito milestone run 03 --part 1  # Run Part 1 only[/dim]\n"
                "[dim]  tito milestone run 03 --part 2  # Run Part 2 only[/dim]\n"
                "[dim]  tito milestone info 03[/dim]\n"
                "[dim]  tito milestone status --detailed[/dim]",
                title="🏆 Milestone System",
                border_style="bright_cyan"
            ))
            return 0

        # Execute the appropriate subcommand
        if args.milestone_command == 'list':
            return self._handle_list_command(args)
        elif args.milestone_command == 'run':
            return self._handle_run_command(args)
        elif args.milestone_command == 'info':
            return self._handle_info_command(args)
        elif args.milestone_command == 'status':
            return self._handle_status_command(args)
        elif args.milestone_command == 'timeline':
            return self._handle_timeline_command(args)
        elif args.milestone_command == 'test':
            return self._handle_test_command(args)
        elif args.milestone_command == 'demo':
            return self._handle_demo_command(args)
        else:
            console.print(Panel(
                f"[red]Unknown milestone subcommand: {args.milestone_command}[/red]",
                title="Error",
                border_style="red"
            ))
            return 1

    def _handle_status_command(self, args: Namespace) -> int:
        """Handle milestone status command."""
        console = self.console
        milestone_system = MilestoneSystem(self.config)
        status = milestone_system.get_milestone_status()

        # Show header with overall progress. Note: status['overall_progress']
        # is unlock-based (it also drives the timeline progress bar elsewhere,
        # where that's the correct meaning), so it isn't used here: showing
        # it next to "Milestones Achieved" would be contradictory (e.g. "0/6
        # achieved" beside "100%"). This header's percentage is achievement-
        # based instead, to actually match the achieved count shown above it.
        total_milestones = len(milestone_system.MILESTONES)
        achievement_progress = (status['total_completed'] / total_milestones) * 100 if total_milestones > 0 else 0
        console.print(Panel(
            f"[bold cyan]🎮 TinyTorch Milestone Progress[/bold cyan]\n\n"
            f"[bold]Capabilities Unlocked:[/bold] {status['total_unlocked']}/{total_milestones} milestones\n"
            f"[bold]Milestones Achieved:[/bold] {status['total_completed']}/{total_milestones} milestones\n"
            f"[bold]Overall Progress:[/bold] {achievement_progress:.0f}%\n\n"
            f"[dim]Transform from student to ML Systems Engineer![/dim]",
            title="🚀 Your Epic Journey",
            border_style="bright_blue"
        ))

        # Show milestone status
        for milestone_id in sorted(milestone_system.MILESTONES.keys()):
            milestone = status["milestones"][milestone_id]
            self._show_milestone_status(milestone, args.detailed)

        # Show next steps
        if status["next_milestone"]:
            next_milestone = status["milestones"][status["next_milestone"]]
            console.print(Panel(
                f"[bold cyan]🎯 Next Achievement[/bold cyan]\n\n"
                f"[bold yellow]{next_milestone['emoji']} {next_milestone['title']}[/bold yellow]\n"
                f"[dim]{next_milestone['victory_condition']}[/dim]\n\n"
                f"[green]Ready to run![/green]\n"
                f"[dim]tito milestone run {next_milestone['id']}[/dim]",
                title="Next Milestone",
                border_style="bright_green"
            ))
        elif status["total_completed"] == total_milestones:
            console.print(Panel(
                f"[bold green]🏆 QUEST COMPLETE! 🏆[/bold green]\n\n"
                f"[green]You've achieved all {total_milestones} epic milestones![/green]\n"
                f"[bold white]You are now an ML Systems Engineer![/bold white]\n\n"
                f"[cyan]Share your achievement and inspire others![/cyan]",
                title="🌟 FULL MASTERY ACHIEVED",
                border_style="bright_green"
            ))
        elif status["total_unlocked"] == total_milestones:
            console.print(Panel(
                f"[bold yellow]⚡ All milestones unlocked![/bold yellow]\n\n"
                f"[yellow]Every milestone is ready to run.[/yellow]\n"
                f"[dim]Run each with tito milestone run <id> to actually achieve it.[/dim]",
                title="🔓 All Milestones Ready",
                border_style="bright_yellow"
            ))

        return 0

    def _show_milestone_status(self, milestone: dict, detailed: bool = False) -> None:
        """Show status for a single milestone."""
        console = self.console

        # Status indicator
        if milestone["is_completed"]:
            status_icon = "✅"
            status_color = "bold green"
            status_text = "ACHIEVED"
        elif milestone["is_unlocked"]:
            status_icon = "🔓"
            status_color = "green"
            status_text = "UNLOCKED"
        elif milestone["can_unlock"]:
            status_icon = "⚡"
            status_color = "yellow"
            status_text = "READY TO UNLOCK"
        elif milestone["required_complete"] and not milestone["trigger_complete"]:
            status_icon = "🔒"
            status_color = "cyan"
            if milestone["trigger_module"]:
                status_text = f"COMPLETE: {milestone['trigger_module']}"
            else:
                status_text = "READY"
        else:
            status_icon = "🔒"
            status_color = "dim"
            status_text = "LOCKED"

        # Basic display
        milestone_content = (
            f"[{status_color}]{status_icon} {milestone['emoji']} {milestone['title']}[/{status_color}]\n"
            f"[dim]{milestone['victory_condition']}[/dim]"
        )

        # Add detailed information if requested
        if detailed:
            req_status = "✅" if milestone["required_complete"] else "❌"
            if milestone["trigger_module"]:
                trigger_status = "✅" if milestone["trigger_complete"] else "❌"
                trigger_text = milestone["trigger_module"]
            else:
                trigger_status = "•"
                trigger_text = "N/A"

            required_modules_str = ', '.join(f"{m:02d}" for m in milestone.get('required_modules', []))

            milestone_content += (
                f"\n\n[bold]Requirements:[/bold]\n"
                f"  {req_status} Modules: {required_modules_str}\n"
                f"  {trigger_status} Trigger: {trigger_text}\n"
                f"[bold]Capability:[/bold] {milestone['capability']}\n"
                f"[bold]Impact:[/bold] {milestone['real_world_impact']}"
            )

            if milestone["is_unlocked"] and milestone.get("unlock_date"):
                unlock_date = datetime.fromisoformat(milestone["unlock_date"]).strftime("%Y-%m-%d")
                milestone_content += f"\n[dim]Unlocked: {unlock_date}[/dim]"

        console.print(Panel(
            milestone_content,
            title=f"Milestone {milestone['id']}",
            border_style=status_color
        ))

    def _handle_timeline_command(self, args: Namespace) -> int:
        """Handle milestone timeline command."""
        console = self.console
        milestone_system = MilestoneSystem(self.config)
        status = milestone_system.get_milestone_status()

        if args.horizontal:
            self._show_horizontal_timeline(status, milestone_system)
        else:
            self._show_tree_timeline(status, milestone_system)

        return 0

    def _show_horizontal_timeline(self, status: dict, milestone_system: MilestoneSystem) -> None:
        """Show horizontal progress bar timeline."""
        console = self.console

        total_milestones = len(milestone_system.MILESTONES)
        console.print(Panel(
            f"[bold cyan]🎮 Milestone Timeline[/bold cyan]\n\n"
            f"[bold]Progress:[/bold] {status['total_unlocked']}/{total_milestones} milestones unlocked",
            title="Your Epic Journey",
            border_style="bright_blue"
        ))

        # Create progress bar
        progress_width = 50
        total_milestones = len(milestone_system.MILESTONES)
        unlocked_width = int((status["total_unlocked"] / total_milestones) * progress_width)

        # Create milestone markers
        timeline = []
        for milestone_id in sorted(milestone_system.MILESTONES.keys()):
            milestone = status["milestones"][milestone_id]

            if milestone["is_unlocked"]:
                marker = f"[green]{milestone['emoji']}[/green]"
            elif milestone["can_unlock"]:
                marker = f"[yellow blink]{milestone['emoji']}[/yellow blink]"
            else:
                marker = f"[dim]{milestone['emoji']}[/dim]"

            timeline.append(marker)

        # Show timeline
        console.print(f"\n{'  '.join(timeline)}")

        # Progress bar
        filled = "█" * unlocked_width
        empty = "░" * (progress_width - unlocked_width)
        console.print(f"\n[green]{filled}[/green][dim]{empty}[/dim]")
        console.print(f"[dim]{status['overall_progress']:.0f}% complete[/dim]\n")

    def _show_tree_timeline(self, status: dict, milestone_system: MilestoneSystem) -> None:
        """Show tree-style milestone timeline."""
        console = self.console

        console.print(Panel(
            f"[bold cyan]🎮 Milestone Progression Tree[/bold cyan]\n\n"
            f"[bold]Your journey from student to ML Systems Engineer[/bold]",
            title="Epic Timeline",
            border_style="bright_blue"
        ))

        # Create tree structure
        tree = Tree("🚀 [bold]TinyTorch Mastery Journey[/bold]")

        for milestone_id in sorted(milestone_system.MILESTONES.keys()):
            milestone = status["milestones"][milestone_id]

            if milestone["is_unlocked"]:
                node_style = "green"
                icon = "✅"
            elif milestone["can_unlock"]:
                node_style = "yellow"
                icon = "⚡"
            else:
                node_style = "dim"
                icon = "🔒"

            branch = tree.add(
                f"[{node_style}]{icon} {milestone['emoji']} {milestone['title']}[/{node_style}]"
            )

            # Add capability description
            branch.add(f"[dim]{milestone['capability']}[/dim]")

            # Add trigger module info
            if not milestone["trigger_module"]:
                required_modules_str = ', '.join(f"{m:02d}" for m in milestone.get('required_modules', []))
                if milestone["required_complete"]:
                    branch.add(f"[green]✅ Prerequisites complete: {required_modules_str}[/green]")
                else:
                    branch.add(f"[dim]🎯 Complete modules: {required_modules_str}[/dim]")
            elif milestone["trigger_complete"]:
                branch.add(f"[green]✅ {milestone['trigger_module']} completed[/green]")
            else:
                branch.add(f"[dim]🎯 Complete: {milestone['trigger_module']}[/dim]")

        console.print(tree)
        console.print()

    def _handle_test_command(self, args: Namespace) -> int:
        """Handle milestone test command."""
        console = self.console
        milestone_system = MilestoneSystem(self.config)

        # Determine which milestone to test
        if args.milestone_id:
            milestone_id = args.milestone_id
        else:
            # Test next available milestone
            status = milestone_system.get_milestone_status()
            if status["next_milestone"]:
                milestone_id = status["next_milestone"]
            else:
                console.print(Panel(
                    "[yellow]No milestone available to test.[/yellow]\n\n"
                    "Either all milestones are unlocked or none are ready.\n"
                    "Use [dim]tito milestone status[/dim] to see your progress.",
                    title="No Test Available",
                    border_style="yellow"
                ))
                return 0

        # Validate milestone ID
        if milestone_id not in milestone_system.MILESTONES:
            console.print(Panel(
                f"[red]Invalid milestone ID: {milestone_id}[/red]\n\n"
                f"Valid milestone IDs: 1, 2, 3, 4, 5, 6",
                title="Invalid Milestone",
                border_style="red"
            ))
            return 1

        milestone = milestone_system.MILESTONES[milestone_id]

        console.print(Panel(
            f"[bold cyan]🧪 Testing Milestone {milestone_id}[/bold cyan]\n\n"
            f"[bold]{milestone['emoji']} {milestone['title']}[/bold]\n"
            f"[dim]{milestone.get('victory_condition', milestone.get('description', ''))}[/dim]",
            title="Milestone Test",
            border_style="bright_cyan"
        ))

        # Run the test with progress animation
        with console.status(f"[bold green]Testing milestone requirements...", spinner="dots"):
            result = milestone_system.run_milestone_test(milestone_id)

        # Show results
        if result["success"]:
            console.print(Panel(
                f"[bold green]✅ Milestone Test Passed![/bold green]\n\n"
                f"[green]All requirements met for {result['milestone_name']}[/green]\n"
                f"[cyan]Capability: {result['capability']}[/cyan]\n\n"
                f"[bold yellow]Run the milestone:[/bold yellow]\n"
                f"[dim]tito milestone run {milestone_id}[/dim]",
                title="🎉 Ready to Unlock!",
                border_style="green"
            ))
        else:
            console.print(Panel(
                f"[bold yellow]⚠️ Milestone Requirements Not Met[/bold yellow]\n\n"
                f"[yellow]Milestone: {result.get('milestone_name', 'Unknown')}[/yellow]\n"
                f"[red]Issue: {result.get('error', 'Unknown error')}[/red]\n\n"
                f"[cyan]Complete the required modules and try again.[/cyan]",
                title="Requirements Missing",
                border_style="yellow"
            ))
            return 1  # 2026-09-29: returned 0 even when requirements were missing

        return 0

    def _handle_demo_command(self, args: Namespace) -> int:
        """Handle milestone demo command."""
        console = self.console
        milestone_system = MilestoneSystem(self.config)
        milestone_id = args.milestone_id

        # Validate milestone ID
        if milestone_id not in milestone_system.MILESTONES:
            console.print(Panel(
                f"[red]Invalid milestone ID: {milestone_id}[/red]\n\n"
                f"Valid milestone IDs: 1, 2, 3, 4, 5, 6",
                title="Invalid Milestone",
                border_style="red"
            ))
            return 1

        milestone = milestone_system.MILESTONES[milestone_id]
        status = milestone_system.get_milestone_status()
        milestone_status = status["milestones"][milestone_id]

        # Check if milestone is unlocked
        if not milestone_status["is_unlocked"]:
            console.print(Panel(
                f"[yellow]Milestone {milestone_id} not yet unlocked.[/yellow]\n\n"
                f"[bold]{milestone['emoji']} {milestone['title']}[/bold]\n"
                f"[dim]{milestone.get('victory_condition', milestone.get('description', ''))}[/dim]\n\n"
                f"[cyan]Complete the requirements first:[/cyan]\n"
                f"[dim]tito milestone test {milestone_id}[/dim]",
                title="Milestone Locked",
                border_style="yellow"
            ))
            return 0

        # Check if demo file exists
        demo_file = milestone.get("demo_file")
        if not demo_file:
            console.print(Panel(
                f"[yellow]Demo not available for Milestone {milestone_id}[/yellow]\n\n"
                f"Use [dim]tito milestone run {milestone_id}[/dim] to run the milestone script.",
                title="Demo Unavailable",
                border_style="yellow"
            ))
            return 0

        demo_path = Path("capabilities") / demo_file
        if not demo_path.exists():
            console.print(Panel(
                f"[yellow]Demo not available for Milestone {milestone_id}[/yellow]\n\n"
                f"Demo file not found: {demo_file}\n"
                f"[dim]This demo may be coming in a future update.[/dim]",
                title="Demo Unavailable",
                border_style="yellow"
            ))
            return 0

        # Run the demo
        console.print(Panel(
            f"[bold cyan]🎬 Launching Milestone {milestone_id} Demo[/bold cyan]\n\n"
            f"[bold]{milestone['emoji']} {milestone['title']}[/bold]\n"
            f"[yellow]Watch your capability in action![/yellow]\n\n"
            f"[cyan]Demonstrating: {milestone.get('capability', milestone.get('description', ''))}[/cyan]\n"
            f"[dim]Running: {demo_file}[/dim]",
            title="Capability Demo",
            border_style="bright_cyan"
        ))

        try:
            result = subprocess.run(
                [sys.executable, str(demo_path)],
                capture_output=False,
                text=True, encoding="utf-8", errors="replace"
            )

            if result.returncode == 0:
                console.print(Panel(
                    f"[bold green]✅ Demo completed successfully![/bold green]\n\n"
                    f"[yellow]You've seen your {milestone['title']} capability in action![/yellow]\n"
                    f"[cyan]Real-world impact: {milestone.get('real_world_impact', milestone.get('historical_context', ''))}[/cyan]",
                    title="🎉 Demo Complete",
                    border_style="green"
                ))
            else:
                console.print(f"[yellow]⚠️ Demo completed with status: {result.returncode}[/yellow]")

        except Exception as e:
            console.print(Panel(
                f"[red]❌ Error running demo: {e}[/red]\n\n"
                f"[dim]You can manually run: python capabilities/{demo_file}[/dim]",
                title="Demo Error",
                border_style="red"
            ))
            return 1

        return 0

    def _handle_list_command(self, args: Namespace) -> int:
        """Handle milestone list command - show available milestones."""
        console = self.console

        min_year = min(m["year"] for m in MILESTONE_SCRIPTS.values())
        max_year = max(m["year"] for m in MILESTONE_SCRIPTS.values())
        console.print(Panel(
            "[bold cyan]🏆 TinyTorch Milestones[/bold cyan]\n\n"
            f"[dim]Recreate ML history from {min_year} to {max_year}[/dim]",
            title="Available Milestones",
            border_style="bright_cyan"
        ))

        # Check module completion status from the canonical module progress file.
        completed_module_nums = _load_completed_module_numbers()

        # Check milestone completion
        milestone_progress = self._get_milestone_progress_data()
        completed_milestones = milestone_tracker.completed_ids(milestone_progress)

        for milestone_id in sorted(MILESTONE_SCRIPTS.keys()):
            milestone = MILESTONE_SCRIPTS[milestone_id]
            required_modules = _required_modules_for(milestone)

            # Check if prerequisites met (required_modules contains integers)
            prereqs_met = all(mod in completed_module_nums for mod in required_modules)
            is_complete = milestone_id in completed_milestones

            # Status indicator
            if is_complete:
                status_icon = "✅"
                status_color = "green"
                status_text = "COMPLETE"
            elif prereqs_met:
                status_icon = "🎯"
                status_color = "yellow"
                status_text = "READY TO RUN"
            else:
                status_icon = "🔒"
                status_color = "dim"
                status_text = "LOCKED"

            # Build display
            if args.simple:
                console.print(f"[{status_color}]{status_icon} {milestone['id']} - {milestone['name']}[/{status_color}]")
            else:
                milestone_display = (
                    f"[{status_color}]{status_icon} {milestone['emoji']} {milestone['name']}[/{status_color}]\n"
                    f"[bold]{milestone['title']}[/bold]\n"
                    f"[dim]{milestone['description']}[/dim]\n"
                    f"[dim]Historical: {milestone['historical_context']}[/dim]\n\n"
                )

                if not is_complete and milestone_tracker.passed_parts(milestone_progress, milestone_id):
                    missing_parts = milestone_tracker.missing_required_parts(milestone_progress, milestone_id)
                    milestone_display += (
                        "[yellow]Required parts still missing a pass: "
                        + ", ".join(f"Part {p}" for p in missing_parts) + "[/yellow]\n"
                    )
                if prereqs_met and not is_complete:
                    milestone_display += f"[bold yellow]▶ Run now:[/bold yellow] [cyan]tito milestone run {milestone_id}[/cyan]\n"
                elif not prereqs_met:
                    missing = [f"{m:02d}" for m in required_modules if m not in completed_module_nums]
                    milestone_display += f"[dim]Required: Complete modules {', '.join(missing)}[/dim]\n"

                console.print(Panel(
                    milestone_display.strip(),
                    title=f"Milestone {milestone['id']} ({milestone['year']})",
                    border_style=status_color
                ))

        return 0

    def _handle_run_command(self, args: Namespace) -> int:
        """Handle milestone run command - run a milestone with checks."""
        if getattr(args, 'non_interactive', False):
            os.environ["TINYTORCH_NON_INTERACTIVE"] = "1"
        console = self.console
        milestone_id = args.milestone_id

        # Resolve name aliases (e.g., "perceptron" -> "01")
        if milestone_id.lower() in MILESTONE_ALIASES:
            milestone_id = MILESTONE_ALIASES[milestone_id.lower()]

        # Validate milestone ID
        if milestone_id not in MILESTONE_SCRIPTS:
            alias_list = ', '.join(sorted(MILESTONE_ALIASES.keys()))
            console.print(Panel(
                f"[red]Invalid milestone: {args.milestone_id}[/red]\n\n"
                f"Valid IDs: {', '.join(sorted(MILESTONE_SCRIPTS.keys()))}\n"
                f"Valid names: {alias_list}",
                title="Invalid Milestone",
                border_style="red"
            ))
            return 1

        milestone = MILESTONE_SCRIPTS[milestone_id]

        # Decide which parts to run. Completion is recorded per part (see
        # tito/core/milestone_tracker.py), so the default must run every
        # REQUIRED part: running only Part 1 used to record the whole
        # milestone (2026-09-29 audit, B1). `--part N` runs and records only N.
        n_parts = milestone_tracker.part_count(milestone)
        req_parts = milestone_tracker.required_parts(milestone)
        run_all = getattr(args, "all", False)

        if "scripts" in milestone:
            all_script_configs = milestone["scripts"]
        else:
            all_script_configs = [dict(milestone, name="Main")]  # milestone-level config

        if args.part is not None:
            if run_all:
                console.print("[yellow]Notice: Both --part and --all specified; prioritizing --part[/yellow]\n")
            if n_parts == 1:
                console.print(f"[yellow]Notice: Milestone {milestone_id} has only one part, ignoring --part flag[/yellow]\n")
                selected_parts = [1]
            elif args.part < 1 or args.part > n_parts:
                console.print(Panel(
                    f"[red]Invalid part number: {args.part}[/red]\n\n"
                    f"Milestone {milestone_id} has {n_parts} parts.\n"
                    f"Valid parts: 1-{n_parts}\n\n"
                    f"[dim]Available parts:[/dim]\n" +
                    "\n".join(f"  Part {i+1}: {s['name']} - {s.get('description', '')}"
                              for i, s in enumerate(all_script_configs)),
                    title="Invalid Part",
                    border_style="red"
                ))
                return 1
            else:
                selected_parts = [args.part]
                role = "required" if args.part in req_parts else "optional extension"
                console.print(f"[dim]Running Part {args.part} of {n_parts} ({role}); only this part is recorded[/dim]\n")
        elif run_all:
            if n_parts == 1:
                console.print(f"[yellow]Notice: Milestone {milestone_id} has only one part, ignoring --all flag[/yellow]\n")
            selected_parts = list(range(1, n_parts + 1))
            if n_parts > 1:
                console.print(f"[bold cyan]Running all {n_parts} parts sequentially[/bold cyan]\n")
        else:
            selected_parts = list(req_parts)
            is_interactive = (
                sys.stdin.isatty()
                and sys.stdout.isatty()
                and os.environ.get("TINYTORCH_NON_INTERACTIVE") != "1"
                and os.environ.get("CI") != "true"
            )
            req_text = ", ".join(str(p) for p in req_parts)

            if is_interactive and n_parts > 1:
                menu_lines = []
                for i, s in enumerate(all_script_configs):
                    tag = " (required)" if (i + 1) in req_parts else " (optional)"
                    menu_lines.append(f"  [{i+1}] {s['name']}: {s.get('description', '')}{tag}")
                menu_lines.append("  [A] Run all parts sequentially")
                console.print(Panel(
                    f"[bold cyan]Milestone {milestone_id} Parts Menu:[/bold cyan]\n\n"
                    + "\n".join(menu_lines)
                    + f"\n\n[dim]Press Enter to run the required parts ({req_text})[/dim]",
                    title="Choose Part to Run",
                    border_style="cyan"
                ))
                try:
                    prompt_text = f"[bold yellow]Select part (1 to {n_parts}, A for all, or Enter for required parts {req_text}): [/bold yellow]"
                    user_choice = console.input(prompt_text).strip()
                    if user_choice.lower() in ("a", "all"):
                        selected_parts = list(range(1, n_parts + 1))
                    elif user_choice.isdigit() and 1 <= int(user_choice) <= n_parts:
                        selected_parts = [int(user_choice)]
                    elif user_choice != "":
                        console.print(f"[yellow]Invalid selection '{user_choice}'; running the required parts ({req_text})[/yellow]\n")
                except (EOFError, KeyboardInterrupt):
                    pass
            elif n_parts > 1:
                parts_overview = "\n".join(
                    f"  {'▶' if i+1 in selected_parts else ' '} Part {i+1}: {s['name']}"
                    f" ({'required' if i+1 in req_parts else 'optional'})"
                    for i, s in enumerate(all_script_configs)
                )
                console.print(
                    f"[dim]Milestone {milestone_id} has {n_parts} parts:[/dim]\n"
                    f"[dim]{parts_overview}[/dim]\n"
                    f"[dim]Running the required parts ({req_text}). Use --part N for one part or --all for every part.[/dim]\n"
                )

        script_configs = [all_script_configs[p - 1] for p in selected_parts]
        scripts_to_run = [(c["name"], c["script"], c.get("description", "")) for c in script_configs]
        # Check if all scripts exist
        for script_name, script_file, _ in scripts_to_run:
            script_path = Path(script_file)
            if not script_path.exists():
                console.print(Panel(
                    f"[red]Milestone script not found![/red]\n\n"
                    f"Expected: {script_file}\n"
                    f"[dim]This milestone may not be implemented yet.[/dim]",
                    title="Script Not Found",
                    border_style="red"
                ))
                return 1

        # Check prerequisites and validate exports/tests (unless skipped)
        if args.skip_checks:
            console.print("[yellow]⚠️ --skip-checks: running as a demo. Results will NOT be recorded.[/yellow]\n")
        else:
            console.print(f"\n[bold cyan]🔍 Checking prerequisites for Milestone {milestone_id}...[/bold cyan]\n")

            # Check module completion status using module workflow
            from .module.workflow import ModuleWorkflowCommand

            module_workflow = ModuleWorkflowCommand(self.config)
            progress_data = module_workflow.get_progress_data()

            # Determine required modules based on what we're running
            # If running specific part(s), use per-part requirements if available
            # Otherwise use milestone-level requirements
            required_modules = set()
            for config in script_configs:
                part_reqs = config.get('required_modules', milestone.get('required_modules', []))
                required_modules.update(part_reqs)
            required_modules = sorted(required_modules)

            completed_modules = progress_data.get('completed_modules', [])

            # Convert completed to set of integers. Handles "01" and "01_tensor".
            completed_set = {
                module_num
                for module_num in (_module_progress_to_int(m) for m in completed_modules)
                if module_num is not None
            }
            missing_modules = [m for m in required_modules if m not in completed_set]

            if missing_modules:
                part_info = ""
                if args.part is not None and len(script_configs) == 1:
                    part_info = f" (Part {args.part})"
                console.print(Panel(
                    f"[bold yellow]❌ Missing Required Modules[/bold yellow]\n\n"
                    f"[yellow]Milestone {milestone_id}{part_info} requires modules: {', '.join(f'{m:02d}' for m in required_modules)}[/yellow]\n"
                    f"[red]Missing: {', '.join(f'{m:02d}' for m in missing_modules)}[/red]\n\n"
                    f"[cyan]Complete the missing modules first:[/cyan]\n" +
                    "\n".join(f"[dim]  tito module complete {m:02d}[/dim]" for m in missing_modules[:3]),
                    title="Prerequisites Not Met",
                    border_style="yellow"
                ))
                return 1

            console.print(f"[green]✅ All required modules completed![/green]\n")

            # Test imports work
            console.print("[bold cyan]🧪 Testing YOUR implementations...[/bold cyan]\n")

            sys.path.insert(0, str(Path.cwd()))

            export_failures = _validate_required_exports(required_modules)
            if export_failures:
                console.print(Panel(
                    f"[red]Import Test Failed![/red]\n\n"
                    f"[yellow]Missing or invalid exports:[/yellow]\n"
                    + "\n".join(f"  • {failure}" for failure in export_failures[:8])
                    + ("\n  • ..." if len(export_failures) > 8 else "")
                    + "\n\n"
                    f"[dim]Your modules may not be exported correctly.[/dim]\n"
                    f"[dim]Try re-exporting: tito module complete XX[/dim]",
                    title="Import Test Failed",
                    border_style="red"
                ))
                return 1

            for module_num in required_modules:
                console.print(f"  [green]✓[/green] Module {module_num:02d} exports available")

            _warn_on_stale_exports(console, module_workflow.config.project_root, required_modules)

            console.print(f"\n[green]✅ YOUR Tiny🔥Torch is ready![/green]\n")

        # Show milestone banner
        scripts_info = ""
        if len(scripts_to_run) > 1:
            scripts_info = "[bold]📂 Parts:[/bold]\n" + "\n".join(
                f"  • {name}: {desc}" for name, _, desc in scripts_to_run
            )
        else:
            scripts_info = f"[bold]📂 Running:[/bold] {scripts_to_run[0][1]}"

        line1_text = f"  {milestone['emoji']} Milestone {milestone_id}: {milestone['name']}"
        line2_text = f"  {milestone['title']}"
        # The box grows to fit the longest title (Milestone 06's overflowed a
        # fixed 48 columns, 2026-09-29).
        WIDTH = max(48, cell_len(line1_text) + 2, cell_len(line2_text) + 2)

        line1 = f"[bold magenta]║[/bold magenta]{line1_text}{' ' * (WIDTH - cell_len(line1_text))}[bold magenta]║[/bold magenta]"

        line2 = f"[bold magenta]║[/bold magenta]{line2_text}{' ' * (WIDTH - cell_len(line2_text))}[bold magenta]║[/bold magenta]"

        console.print(Panel(
            f"[bold magenta]╔{'═' * WIDTH}╗[/bold magenta]\n"
            f"{line1}\n"
            f"{line2}\n"
            f"[bold magenta]╚{'═' * WIDTH}╝[/bold magenta]\n\n"
            f"[bold]📚 Historical Context:[/bold]\n"
            f"{milestone['historical_context']}\n\n"
            f"[bold]🎯 What You'll Do:[/bold]\n"
            f"{milestone['description']}\n\n"
            f"{scripts_info}\n\n"
            f"[dim]All code uses YOUR Tiny🔥Torch implementations![/dim]",
            title=f"🏆 Milestone {milestone_id} ({milestone['year']})",
            border_style="bright_magenta",
            padding=(1, 2)
        ))

        # Only prompt if in interactive terminal and not non-interactive mode
        if sys.stdin.isatty() and sys.stdout.isatty() and os.environ.get("TINYTORCH_NON_INTERACTIVE") != "1" and os.environ.get("CI") != "true":
            try:
                console.input("\n[yellow]Press Enter to begin...[/yellow] ")
            except EOFError:
                pass

        # Run the selected parts, recording each part's outcome as it finishes.
        tito_dir = _tito_dir()
        part_outcomes = []  # (part_no, name, passed)
        interrupted = False
        for idx, (part_no, (script_name, script_file, script_desc)) in enumerate(zip(selected_parts, scripts_to_run)):
            if len(scripts_to_run) > 1 or n_parts > 1:
                console.print(f"\n[bold cyan]━━━ Part {part_no}/{n_parts}: {script_name} ━━━[/bold cyan]")
                if script_desc:
                    console.print(f"[dim]{script_desc}[/dim]\n")
            else:
                console.print(f"\n[bold green]🚀 Starting Milestone {milestone_id}...[/bold green]\n")

            console.print("━" * 80 + "\n")

            try:
                returncode = run_milestone_script(script_file)
            except Exception as e:
                console.print(f"[red]Error running {script_name}: {e}[/red]")
                returncode = 1
            if returncode in INTERRUPTED_EXIT_CODES:
                console.print(f"\n\n[yellow]⚠️ Milestone interrupted by user (Part {part_no} not recorded)[/yellow]")
                interrupted = True
                break

            console.print("\n" + "━" * 80)
            passed = returncode == 0
            part_outcomes.append((part_no, script_name, passed))
            if not args.skip_checks:
                milestone_tracker.record_part_result(tito_dir, milestone_id, part_no, passed)

            if not passed:
                console.print(f"[yellow]⚠️ Part {part_no} ({script_name}) failed (exit code {returncode})[/yellow]")
                if idx < len(scripts_to_run) - 1:
                    # Ask to continue only in an interactive terminal; otherwise stop.
                    if sys.stdin.isatty() and sys.stdout.isatty() and os.environ.get("TINYTORCH_NON_INTERACTIVE") != "1":
                        try:
                            cont = console.input("\n[yellow]Continue to next part? (y/n): [/yellow] ")
                        except EOFError:
                            cont = "n"
                        if cont.strip().lower() != "y":
                            break
                    else:
                        break

        all_passed = (not interrupted) and len(part_outcomes) == len(scripts_to_run) \
            and all(ok for _, _, ok in part_outcomes)

        if args.skip_checks:
            console.print(Panel(
                "[yellow]--skip-checks ran this milestone as a demo. Nothing was recorded:[/yellow]\n"
                "[yellow]no part result and no milestone completion.[/yellow]\n\n"
                f"[dim]Run without --skip-checks to record progress: tito milestone run {milestone_id}[/dim]",
                title="Not Recorded",
                border_style="yellow"
            ))
            if interrupted:
                return 130
            return 0 if all_passed else 1

        if interrupted:
            return 130

        ledger = milestone_tracker.load(tito_dir)
        missing = milestone_tracker.missing_required_parts(ledger, milestone_id)

        if all_passed and not missing:
            parts_text = ""
            if n_parts > 1:
                passed_now = {p for p, _, ok in part_outcomes if ok}
                parts_text = "\n\n[bold]Required parts passed:[/bold]\n" + "\n".join(
                    f"  ✅ Part {p}: {milestone_tracker.part_name(milestone, p)}"
                    + ("" if p in passed_now else " [dim](earlier run)[/dim]")
                    for p in req_parts
                )

            default_highlights = [
                "Every line of code: YOUR implementations",
                "Every tensor operation: YOUR Tensor class",
                "Every gradient: YOUR autograd",
            ]
            highlights = MILESTONE_ACHIEVEMENT_HIGHLIGHTS.get(milestone_id, default_highlights)
            highlights_text = "\n".join(f"• {line}" for line in highlights)

            console.print(Panel(
                f"[bold green]🏆 MILESTONE ACHIEVED![/bold green]\n\n"
                f"[green]You completed Milestone {milestone_id}: {milestone['name']}[/green]\n"
                f"[yellow]{milestone['title']}[/yellow]{parts_text}\n\n"
                f"[bold]What makes this special:[/bold]\n"
                f"{highlights_text}\n\n"
                f"[cyan]Achievement saved locally![/cyan]",
                title="✨ Achievement Unlocked ✨",
                border_style="bright_green",
                padding=(1, 2)
            ))

            # Offer to sync progress (uses centralized SubmissionHandler)
            self._offer_progress_sync(milestone_id, milestone['name'])

            # Show next steps
            next_id = str(int(milestone_id) + 1).zfill(2)
            if next_id in MILESTONE_SCRIPTS:
                next_milestone = MILESTONE_SCRIPTS[next_id]
                console.print(f"\n[bold yellow]🎯 What's Next:[/bold yellow]")
                console.print(f"[dim]Milestone {next_id}: {next_milestone['name']}[/dim]")
                completed_modules = _load_completed_module_numbers()
                missing_mods = [m for m in next_milestone["required_modules"] if m not in completed_modules]
                if missing_mods:
                    console.print(f"[dim]Unlock by completing modules: {', '.join(f'{m:02d}' for m in missing_mods[:3])}[/dim]")
                else:
                    console.print(f"[green]Ready to run: tito milestone run {next_id}[/green]")
            return 0

        # Not complete: say exactly which parts ran, which passed, and what is missing.
        lines = []
        for p, name, ok in part_outcomes:
            mark = "[green]✅ passed[/green]" if ok else "[red]❌ failed[/red]"
            lines.append(f"  Part {p}: {name}: {mark} (recorded)")
        for p in selected_parts[len(part_outcomes):]:
            lines.append(f"  Part {p}: {milestone_tracker.part_name(milestone, p)}: [dim]not run[/dim]")
        if missing:
            missing_text = ", ".join(f"Part {p} ({milestone_tracker.part_name(milestone, p)})" for p in missing)
            status_line = (
                f"[bold yellow]Milestone {milestone_id} is NOT complete.[/bold yellow]\n"
                f"[yellow]Required parts still missing a pass: {missing_text}[/yellow]"
            )
            if len(missing) == 1:
                hint = f"tito milestone run {milestone_id} --part {missing[0]}" if n_parts > 1 else f"tito milestone run {milestone_id}"
            else:
                hint = f"tito milestone run {milestone_id}"
            status_line += f"\n\n[dim]Next: {hint}[/dim]"
        else:
            status_line = f"[green]Milestone {milestone_id} stays complete from earlier runs.[/green]"
        console.print(Panel(
            "\n".join(lines) + "\n\n" + status_line,
            title=f"Milestone {milestone_id} Parts",
            border_style="yellow" if missing or not all_passed else "green"
        ))
        return 0 if all_passed else 1

    def _handle_info_command(self, args: Namespace) -> int:
        """Handle milestone info command - show detailed information."""
        console = self.console
        milestone_id = args.milestone_id

        # Resolve name aliases (e.g., "perceptron" -> "01")
        if milestone_id.lower() in MILESTONE_ALIASES:
            milestone_id = MILESTONE_ALIASES[milestone_id.lower()]

        if milestone_id not in MILESTONE_SCRIPTS:
            alias_list = ', '.join(sorted(MILESTONE_ALIASES.keys()))
            console.print(Panel(
                f"[red]Invalid milestone: {args.milestone_id}[/red]\n\n"
                f"Valid IDs: {', '.join(sorted(MILESTONE_SCRIPTS.keys()))}\n"
                f"Valid names: {alias_list}",
                title="Invalid Milestone",
                border_style="red"
            ))
            return 1

        milestone = MILESTONE_SCRIPTS[milestone_id]

        # Check status
        completed_module_nums = _load_completed_module_numbers()

        prereqs_met = all(m in completed_module_nums for m in milestone["required_modules"])

        # Display detailed info
        info_text = (
            f"[bold cyan]{milestone['emoji']} {milestone['name']}[/bold cyan]\n\n"
            f"[bold]{milestone['title']}[/bold]\n\n"
            f"[yellow]📚 Historical Context:[/yellow]\n"
            f"{milestone['historical_context']}\n\n"
            f"[yellow]🎯 Description:[/yellow]\n"
            f"{milestone['description']}\n\n"
            f"[yellow]📋 Required Modules:[/yellow]\n"
        )

        for mod in milestone["required_modules"]:
            mod_str = f"{mod:02d}"
            if mod in completed_module_nums:
                info_text += f"  [green]✓[/green] Module {mod_str}\n"
            else:
                info_text += f"  [red]✗[/red] Module {mod_str}\n"

        # Show scripts
        if "scripts" in milestone:
            info_text += f"\n[yellow]📂 Scripts ({len(milestone['scripts'])} parts):[/yellow]\n"
            for s in milestone["scripts"]:
                info_text += f"  • {s['name']}: {s['script']}\n"
        else:
            info_text += f"\n[yellow]📂 Script:[/yellow] {milestone['script']}\n"

        if prereqs_met:
            info_text += f"\n[bold green]✅ Ready to run![/bold green]\n[cyan]tito milestone run {milestone_id}[/cyan]"
        else:
            missing = [m for m in milestone["required_modules"] if m not in completed_module_nums]
            info_text += f"\n[bold yellow]🔒 Locked[/bold yellow]\nComplete modules: {', '.join(f'{m:02d}' for m in missing)}"

        console.print(Panel(
            info_text,
            title=f"Milestone {milestone_id} Information",
            border_style="bright_cyan",
            padding=(1, 2)
        ))

        return 0

    def _get_milestone_progress_data(self) -> dict:
        """Read the milestone ledger (normalized; see tito/core/milestone_tracker.py)."""
        return milestone_tracker.load(_tito_dir())

    def _offer_progress_sync(self, milestone_id: str, milestone_name: str) -> None:
        """Offer to sync progress after milestone completion.

        Delegates to the shared :func:`auto_sync_after_completion` helper so the
        CI / interactivity / logged-in rules match the module-completion path.
        Crucially, this no longer skips the sync on a non-TTY shell (Git Bash /
        IDE terminals): that silent skip left progress unsynced (#1849).
        """
        from ..core.submission import auto_sync_after_completion

        self.console.print()
        try:
            auto_sync_after_completion(
                self.config,
                self.console,
                prompt="Sync this achievement to your profile?",
            )
        except Exception as e:
            self.console.print(f"[yellow]⚠️ Could not sync: {e}[/yellow]")
            self.console.print("[dim]Your progress is saved locally and will sync next time.[/dim]")
