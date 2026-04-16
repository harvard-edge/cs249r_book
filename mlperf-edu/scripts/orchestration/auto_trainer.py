"""
MLPerf EDU: Autonomous Multiprocessing Training Factory.

Spawns one process per model, trains on frozen synthetic tensors,
detects convergence plateaus, and optionally invokes the Dual-Agent
LLM researcher to rewrite the architecture.
"""

import importlib
import json
import multiprocessing
import os
import sys
import time
import torch
import torch.optim as optim

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Ensure project root is importable
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PLATEAU_WINDOW = 20       # Epochs to look back for plateau detection
PLATEAU_EPSILON = 1e-4    # Min mean improvement to NOT be a plateau
MAX_NAS_ATTEMPTS = 3      # Max LLM rewrite cycles before giving up
MAX_EPOCHS = 350          # Hard cap per NAS attempt
CHECKPOINT_ROOT = "checkpoints"

# ---------------------------------------------------------------------------
# Model Registry — maps workload names to (module_path, class_name)
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "nano-moe-12m":         ("reference.cloud.nano_moe",          "NanoMoEWhiteBox"),
    "nanogpt-12m":          ("reference.cloud.nanogpt_train",     "NanoGPTWhiteBox"),
    "resnet18":             ("reference.edge.resnet_train",       "ResNet18WhiteBox"),
    "micro-dlrm-1m":       ("reference.cloud.micro_dlrm",        "MicroDLRMWhiteBox"),
    "micro-diffusion-32px": ("reference.cloud.micro_diffusion",   "MicroDiffusionUNet"),
    # Tiny division
    "dscnn-kws":            ("reference.tiny.dscnn_kws",          "DSCNN"),
    "anomaly-ae":           ("reference.tiny.anomaly_detection_ae", "AnomalyDetectionAE"),
    # Agent workloads
    "nano-rag-agent":       ("reference.cloud.nano_rag_agent",      "NanoRAGAgent"),
    "nano-codegen-agent":   ("reference.cloud.nano_codegen_agent",  "NanoCodeGenAgent"),
    "nano-react-agent":     ("reference.cloud.nano_react_agent",    "NanoReActAgent"),
    "nano-toolcall-agent":  ("reference.cloud.nano_toolcall_agent", "NanoToolCallAgent"),
}

# Model-specific constructor kwargs
MODEL_KWARGS = {
    "resnet18": {"num_classes": 100},
    "anomaly-ae": {"input_dim": 784, "bottleneck_dim": 8},
}


def _load_model(model_name: str, device: str):
    """Dynamically import and instantiate a model from the registry."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")

    module_path, class_name = MODEL_REGISTRY[model_name]
    # Force a fresh import (needed after LLM rewrites the source file)
    mod = importlib.import_module(module_path)
    mod = importlib.reload(mod)
    cls = getattr(mod, class_name)
    kwargs = MODEL_KWARGS.get(model_name, {})
    return cls(**kwargs).to(device)


def _create_frozen_batch(model_name: str, device: str, batch_size: int = 16):
    """
    Create a fixed batch of synthetic data outside the gradient loop.
    The model must memorize this exact structure — if the architecture is
    broken (missing nonlinearities, shape mismatches), the loss diverges.
    """
    if "resnet" in model_name or "diffusion" in model_name:
        data = torch.randn(batch_size, 3, 32, 32, device=device)
        if "resnet" in model_name:
            target = torch.randint(0, 100, (batch_size,), device=device)
        else:
            target = torch.randn(batch_size, 3, 32, 32, device=device)
    elif "dlrm" in model_name:
        data = {
            "dense_x": torch.randn(batch_size, 16, device=device),
            "sparse_indices": [torch.randint(0, 100, (batch_size,), device=device) for _ in range(3)],
            "sparse_offsets": [torch.arange(0, batch_size, device=device) for _ in range(3)],
        }
        target = torch.rand(batch_size, 1, device=device)
    elif "rag" in model_name:
        # RAG agent: query tokens in, targets for the augmented context
        data = torch.randint(0, 50257, (4, 32), device=device)
        target = torch.randint(0, 50257, (4, 128), device=device)
    elif "codegen" in model_name:
        # CodeGen agent: code tokens in, code tokens out
        data = torch.randint(0, 50257, (4, 64), device=device)
        target = torch.randint(0, 50257, (4, 64), device=device)
    else:
        # Language models (NanoGPT, Nano-MoE)
        data = torch.randint(0, 50257, (4, 64), device=device)
        target = torch.randint(0, 50257, (4, 64), device=device)

    return data, target


def _forward_pass(model, model_name, data, target):
    """Run one forward pass and compute loss. Returns (outputs, loss)."""
    if "resnet" in model_name:
        outputs = model(data)
        loss = torch.nn.functional.cross_entropy(outputs, target)
    elif "diffusion" in model_name:
        outputs = model(data)
        loss = torch.nn.functional.mse_loss(outputs, target)
    elif "dlrm" in model_name:
        outputs = model(data["dense_x"], data["sparse_indices"], data["sparse_offsets"])
        loss = torch.nn.functional.binary_cross_entropy(outputs, target)
    else:
        # Language models return (logits, loss)
        _, loss = model(data, targets=target)
    return loss


def _detect_plateau(loss_history: list) -> bool:
    """Check if loss has plateaued over the last PLATEAU_WINDOW epochs."""
    if len(loss_history) < PLATEAU_WINDOW:
        return False
    window = loss_history[-PLATEAU_WINDOW:]
    mean_improvement = window[0] - window[-1]
    return mean_improvement < PLATEAU_EPSILON


def _invoke_llm_researcher(model_name: str, loss_history: list, val_loss_history: list,
                           target_loss: float, iteration: int) -> bool:
    """
    Call the Dual-Agent LLM loop to rewrite the model architecture.
    Returns True if a new architecture was written and is ready for re-import.
    """
    try:
        from scripts.orchestration.llm_researcher import UltimateAutonomousScientist

        module_path, _ = MODEL_REGISTRY[model_name]
        source_file = module_path.replace(".", "/") + ".py"

        agent = UltimateAutonomousScientist(source_file, model_name)

        # Pass the last 10 loss values as context
        recent_train = loss_history[-10:]
        recent_val = val_loss_history[-10:]

        success = agent.execute_research_loop(
            loss_delta=recent_train,
            validation_accuracy=recent_val,
            target_loss=target_loss,
            iteration=iteration
        )
        return success
    except Exception as e:
        print(f"[{model_name}] ⚠️ LLM Researcher failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Main Training Loop (runs in a subprocess)
# ---------------------------------------------------------------------------

def _training_process(model_name: str, target_loss: float):
    """
    Independent PyTorch training process.
    Trains on REAL data from the dataset factory, detects plateaus, and
    optionally invokes the Dual-Agent NAS loop to rewrite the architecture.
    """
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    print(f"[{model_name}] Initializing on device={device}")

    # Load real data via the dataset factory
    try:
        from reference.dataset_factory import get_dataloaders
        batch_size = 64 if "resnet" in model_name or "diffusion" in model_name else 16
        train_loader, val_loader = get_dataloaders(model_name, batch_size=batch_size)
        print(f"[{model_name}] ✅ Loaded real dataset: "
              f"{len(train_loader.dataset)} train / {len(val_loader.dataset)} val samples")
    except Exception as e:
        print(f"[{model_name}] ⚠️ Dataset factory failed ({e}), falling back to frozen batch")
        train_loader = None
        val_loader = None

    nas_attempt = 0
    total_epochs = 0
    loss_history = []
    val_loss_history = []

    while nas_attempt <= MAX_NAS_ATTEMPTS:
        # (Re-)load the model
        try:
            model = _load_model(model_name, device)
        except Exception as e:
            print(f"[{model_name}] ❌ Failed to load model: {e}")
            break

        model.train()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        epoch = 0

        print(f"[{model_name}] Starting training (NAS attempt {nas_attempt}/{MAX_NAS_ATTEMPTS})")

        while epoch < MAX_EPOCHS:
            # --- Training pass: iterate through real data ---
            epoch_losses = []

            if train_loader is not None:
                for batch_idx, batch in enumerate(train_loader):
                    # Unpack batch based on model type
                    if "dlrm" in model_name:
                        dense, sparse_idx, sparse_off, labels = batch
                        dense = dense.to(device)
                        sparse_idx = [s.to(device) for s in sparse_idx]
                        sparse_off = [s.to(device) for s in sparse_off]
                        labels = labels.to(device)
                        model.train()
                        optimizer.zero_grad()
                        outputs = model(dense, sparse_idx, sparse_off)
                        loss = torch.nn.functional.binary_cross_entropy(outputs, labels)
                    elif "resnet" in model_name:
                        data_batch, target_batch = batch
                        data_batch = data_batch.to(device)
                        target_batch = target_batch.to(device)
                        model.train()
                        optimizer.zero_grad()
                        outputs = model(data_batch)
                        loss = torch.nn.functional.cross_entropy(outputs, target_batch)
                    elif "diffusion" in model_name:
                        data_batch, _ = batch  # CIFAR-10: ignore class labels
                        data_batch = data_batch.to(device)
                        target_batch = data_batch.clone()  # Denoise: reconstruct input
                        model.train()
                        optimizer.zero_grad()
                        outputs = model(data_batch)
                        loss = torch.nn.functional.mse_loss(outputs, target_batch)
                    else:
                        # Language models and agent models: (input_ids, targets)
                        data_batch, target_batch = batch
                        data_batch = data_batch.to(device)
                        target_batch = target_batch.to(device)
                        model.train()
                        optimizer.zero_grad()
                        _, loss = model(data_batch, targets=target_batch)

                    loss.backward()
                    optimizer.step()
                    epoch_losses.append(loss.item())

                    # Cap batches per epoch for speed (especially CIFAR)
                    if batch_idx >= 50:
                        break
            else:
                # Fallback: single frozen batch
                train_data, train_target = _create_frozen_batch(model_name, device)
                optimizer.zero_grad()
                loss = _forward_pass(model, model_name, train_data, train_target)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            train_loss = sum(epoch_losses) / len(epoch_losses)
            loss_history.append(train_loss)

            # --- Validation pass ---
            model.eval()
            val_losses = []
            with torch.no_grad():
                if val_loader is not None:
                    for batch_idx, batch in enumerate(val_loader):
                        if "dlrm" in model_name:
                            dense, sparse_idx, sparse_off, labels = batch
                            dense = dense.to(device)
                            sparse_idx = [s.to(device) for s in sparse_idx]
                            sparse_off = [s.to(device) for s in sparse_off]
                            labels = labels.to(device)
                            outputs = model(dense, sparse_idx, sparse_off)
                            v_loss = torch.nn.functional.binary_cross_entropy(outputs, labels)
                        elif "resnet" in model_name:
                            data_batch, target_batch = batch
                            outputs = model(data_batch.to(device))
                            v_loss = torch.nn.functional.cross_entropy(outputs, target_batch.to(device))
                        elif "diffusion" in model_name:
                            data_batch, _ = batch
                            data_batch = data_batch.to(device)
                            outputs = model(data_batch)
                            v_loss = torch.nn.functional.mse_loss(outputs, data_batch)
                        else:
                            data_batch, target_batch = batch
                            _, v_loss = model(data_batch.to(device), targets=target_batch.to(device))
                        val_losses.append(v_loss.item())
                        if batch_idx >= 10:
                            break
                else:
                    val_data, val_target = _create_frozen_batch(model_name, device)
                    v_loss = _forward_pass(model, model_name, val_data, val_target)
                    val_losses.append(v_loss.item())

            val_loss_val = sum(val_losses) / len(val_losses)
            val_loss_history.append(val_loss_val)

            if epoch % 5 == 0:
                print(f"[{model_name}] Epoch {total_epochs + epoch} | "
                      f"Train: {train_loss:.4f} | Val: {val_loss_val:.4f}")

            # Check convergence
            if train_loss <= target_loss:
                print(f"[{model_name}] ✅ Converged at epoch {total_epochs + epoch} "
                      f"(train_loss={train_loss:.4f} <= {target_loss})")
                total_epochs += epoch
                break

            # Check plateau → trigger NAS
            if _detect_plateau(loss_history) and nas_attempt < MAX_NAS_ATTEMPTS:
                print(f"[{model_name}] 📉 Plateau detected at epoch {total_epochs + epoch}. "
                      f"Triggering Dual-Agent NAS (attempt {nas_attempt + 1})...")
                total_epochs += epoch
                success = _invoke_llm_researcher(
                    model_name, loss_history, val_loss_history,
                    target_loss, nas_attempt + 1
                )
                if success:
                    nas_attempt += 1
                    break  # Break inner loop to reload model
                else:
                    print(f"[{model_name}] LLM rewrite failed, continuing training...")
                    nas_attempt += 1

            epoch += 1
        else:
            # Max epochs reached without convergence or plateau trigger
            total_epochs += epoch
            print(f"[{model_name}] ⏱️ Max epochs reached (total={total_epochs})")
            break

        # If we converged, break the outer NAS loop too
        if loss_history and loss_history[-1] <= target_loss:
            break

    # --- Save results ---
    ckpt_dir = os.path.join(CHECKPOINT_ROOT, model_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save model weights
    save_path = os.path.join(ckpt_dir, "instructor_baseline.pt")
    torch.save(model.state_dict(), save_path)

    # Save structured results
    results = {
        "model_name": model_name,
        "device": device,
        "target_loss": target_loss,
        "final_train_loss": loss_history[-1] if loss_history else None,
        "final_val_loss": val_loss_history[-1] if val_loss_history else None,
        "total_epochs": total_epochs,
        "nas_attempts": nas_attempt,
        "converged": loss_history[-1] <= target_loss if loss_history else False,
        "loss_history": loss_history,
        "val_loss_history": val_loss_history,
    }
    results_path = os.path.join(ckpt_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Plot convergence curves
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=loss_history, mode='lines', name='Train Loss'))
        fig.add_trace(go.Scatter(y=val_loss_history, mode='lines',
                                 name='Validation Loss', line=dict(dash='dash')))
        fig.update_layout(
            title=f'Convergence: {model_name}',
            xaxis_title='Epoch',
            yaxis_title='Loss',
            template='plotly_dark'
        )
        plot_path = os.path.join(ckpt_dir, "convergence_curve.html")
        fig.write_html(plot_path)
        print(f"[{model_name}] 📊 Saved: {save_path}, {results_path}, {plot_path}")
    except ImportError:
        print(f"[{model_name}] 📊 Saved: {save_path}, {results_path} (plotly not available)")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class ConvergenceResearcherAgent:
    """Multiprocessing training factory. Spawns one process per workload."""

    def __init__(self):
        self.workloads = [
            # Traditional workloads
            ("nano-moe-12m",          1.25),
            ("nanogpt-12m",           1.25),
            ("resnet18",              0.75),
            ("micro-dlrm-1m",        0.35),
            ("micro-diffusion-32px",  0.40),
            # Agent workloads
            ("nano-rag-agent",        1.25),
            ("nano-codegen-agent",    1.25),
            ("nano-react-agent",      1.25),
            ("nano-toolcall-agent",   1.25),
        ]

    def launch_grid(self):
        """Launch all workloads in parallel subprocesses."""
        print("🚀 Launching multiprocessing training grid...\n")
        multiprocessing.set_start_method('spawn', force=True)

        processes = []
        for model_name, target in self.workloads:
            p = multiprocessing.Process(
                target=_training_process,
                args=(model_name, target)
            )
            processes.append((model_name, p))
            p.start()

        for model_name, p in processes:
            p.join()
            status = "✅" if p.exitcode == 0 else "❌"
            print(f"{status} {model_name} finished (exit code: {p.exitcode})")

        print("\n🏆 Training grid complete. Checkpoints saved to ./checkpoints/")


if __name__ == "__main__":
    agent = ConvergenceResearcherAgent()
    agent.launch_grid()
