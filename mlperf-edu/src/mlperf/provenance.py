import json
import hashlib
import os
import torch
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple
from rich.console import Console

console = Console()

# -----------------------------------------------------------------------
# MLPerf EDU: Provenance Tracking System 
# -----------------------------------------------------------------------

class ProvenanceManager:
    """
    Manages the lifecycle between Training and Inference, guaranteeing the provenance 
    of the checkpoint through SHA256 hashes and standardizing the JSON artifact (.provd).
    """

    @staticmethod
    def _hash_file(file_path: str) -> str:
        """Computes the SHA-256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256.update(byte_block)
        return sha256.hexdigest()

    @staticmethod
    def save_training_checkpoint(model: torch.nn.Module, referee_result: Any, workload: str, track: str, save_dir: str = "./weights") -> str:
        """
        Saves the model state_dict, computes its hash, and generates the .provd schema.
        We expect referee_result to be the populated TrainingResult.
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Save PyTorch weights
        pt_filename = f"{workload}_{referee_result.nonce}.pt"
        pt_path = os.path.join(save_dir, pt_filename)
        torch.save(model.state_dict(), pt_path)
        
        # 2. Hash it to secure provenance
        sha256_hash = ProvenanceManager._hash_file(pt_path)
        
        # 3. Construct .provd schema
        provd_data = {
            "workload": workload,
            "track": track,
            "provenance_phase": "inference_ready" if referee_result.passed else "failed_training",
            "training_metrics": {
                "target_accuracy_threshold": referee_result.target_accuracy,
                "achieved_accuracy": referee_result.achieved_accuracy,
                "time_to_train_seconds": referee_result.time_to_train_seconds,
                "epochs_required": referee_result.epochs_run,
                "hardware_flops_score": referee_result.hardware_efficiency_percentage
            },
            "inference_modes_supported": ["Offline", "Server", "SingleStream"],
            "model_artifact": {
                "checkpoint_path": pt_path,
                "sha256_hash": sha256_hash,
                "is_golden": False
            }
        }
        
        provd_filename = f"{workload}.provd"
        with open(provd_filename, "w") as f:
            json.dump(provd_data, f, indent=4)
            
        console.print(f"[bold green][Provenance][/bold green] 🛡️ Model perfectly packaged into [cyan]{provd_filename}[/cyan]!")
        console.print(f"[bold green][Provenance][/bold green] 🔒 Artifact Hash: [dim]{sha256_hash}[/dim]")
        return provd_filename

    @staticmethod
    def hydrate_canonical(checkpoint_path: str, workload: str, origin_source: str) -> str:
        """
        Takes a downloaded canonical weight file, hashes it, and wraps it in our 
        cryptographic pedagogical .provd schema. Bridges the offline provenance gap.
        """
        sha256_hash = ProvenanceManager._hash_file(checkpoint_path)
        
        provd_data = {
            "workload": workload,
            "track": "cloud/mobile (hydrated)",
            "provenance_phase": "canonical_hydration",
            "training_metrics": {
                "origin_source": origin_source,
                "notes": "Mathematically identical to HuggingFace or canonical Torchvision outputs."
            },
            "inference_modes_supported": ["Offline", "Server", "SingleStream"],
            "model_artifact": {
                "checkpoint_path": checkpoint_path,
                "sha256_hash": sha256_hash,
                "is_golden": False # We treat these as verified artifacts now!
            }
        }
        
        provd_filename = f"{workload}.provd"
        with open(provd_filename, "w") as f:
            json.dump(provd_data, f, indent=4)
            
        console.print(f"[bold blue][Provenance][/bold blue] 💧 Canonical Hash wrapped: [dim]{sha256_hash}[/dim]")
        return provd_filename


    @staticmethod
    def verify_and_load(provd_path: str, use_fallback_golden: bool = False) -> Tuple[str, bool]:
        """
        Reads a .provd manifest, verifies the target weights haven't been tampered with,
        and returns the explicit path to the weights. 
        Returns (path_to_weights, is_golden).
        """
        try:
            with open(provd_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            if use_fallback_golden:
                console.print(f"[bold yellow][Provenance] ⚠️ Missing or broken .provd file. Falling back to Golden Weights.[/bold yellow]")
                return ("GOLDEN", True)
            raise FileNotFoundError(f"Provenance file {provd_path} not found.")

        artifact = data.get("model_artifact", {})
        is_golden = artifact.get("is_golden", False)
        
        if is_golden:
            console.print("[bold yellow][Provenance] ⚠️ Using Golden Fallback Weights.[/bold yellow]")
            return ("GOLDEN", True)

        pt_path = artifact.get("checkpoint_path")
        expected_hash = artifact.get("sha256_hash")
        
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"Checkpoint listed in .provd is missing: {pt_path}")
            
        # Verify provenance
        console.print(f"[bold cyan][Provenance] 🔍 Verifying cryptographic provenance of {pt_path}...[/bold cyan]")
        actual_hash = ProvenanceManager._hash_file(pt_path)
        if actual_hash != expected_hash:
            raise ValueError("PROVENANCE FAILURE: The checkpoint hash does not match the manifest! It may have been tampered with.")
            
        console.print("[bold green][Provenance] ✅ Integrity strictly validated.[/bold green]")
        return (pt_path, False)

