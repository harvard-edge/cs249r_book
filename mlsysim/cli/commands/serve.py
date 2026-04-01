import typer
from typing import Optional
from mlsysim.cli.schemas import _resolve_model, _resolve_hardware
from mlsysim.cli.exceptions import error_shield, ExitCode, exit_with_code
from mlsysim.models.types import TransformerWorkload


def serve_main(
    ctx: typer.Context,
    model: str = typer.Argument(..., help="Model name (e.g. Llama3_8B)"),
    hardware: str = typer.Argument(..., help="Hardware name (e.g. H100)"),
    seq_len: int = typer.Option(2048, "--seq-len", "-s", help="Sequence length (context window)"),
    batch_size: int = typer.Option(1, "--batch-size", "-b", help="Batch size"),
    precision: str = typer.Option("fp16", "--precision", "-p", help="Numerical precision (fp32, fp16, fp8, int8, int4)"),
    efficiency: float = typer.Option(0.5, "--efficiency", "-e", help="Compute efficiency (0.0 to 1.0)"),
):
    """
    **Evaluate LLM serving performance (prefill + decode).**

    Computes Time-To-First-Token (TTFT), Inter-Token Latency (ITL),
    and throughput for a given model on target hardware.

    **Examples:**

    ```bash
    mlsysim serve Llama3_8B H100 --seq-len 4096 --batch-size 8
    mlsysim serve Llama3_70B H100 --precision fp8 --efficiency 0.6
    ```
    """
    output_format = ctx.obj.get("output_format", "text") if ctx.obj else "text"

    with error_shield(output_format=output_format):
        model_obj = _resolve_model(model)
        hw_obj = _resolve_hardware(hardware)

        if not isinstance(model_obj, TransformerWorkload):
            raise ValueError(
                f"Serving analysis requires a TransformerWorkload, but '{model}' is a {type(model_obj).__name__}. "
                "Use 'mlsysim eval' for non-Transformer workloads."
            )

        from mlsysim.core.solver import ServingModel
        solver = ServingModel()
        result = solver.solve(
            model=model_obj,
            hardware=hw_obj,
            seq_len=seq_len,
            batch_size=batch_size,
            precision=precision,
            efficiency=efficiency,
        )

        # System throughput: batch_size tokens produced per decode step
        itl_s = result.itl.to("s").magnitude
        tokens_per_sec = (batch_size / itl_s) if itl_s > 0 else float("inf")

        if output_format == "json":
            import json
            data = {
                "model": model,
                "hardware": hardware,
                "seq_len": seq_len,
                "batch_size": batch_size,
                "precision": precision,
                "efficiency": efficiency,
                "feasible": result.feasible,
                "ttft_ms": float(result.ttft.to("ms").magnitude),
                "itl_ms": float(result.itl.to("ms").magnitude),
                "tokens_per_sec": round(tokens_per_sec, 1),
                "kv_cache_gb": float(result.kv_cache_size.to("GB").magnitude),
                "memory_utilization": round(result.memory_utilization, 3),
            }
            print(json.dumps(data, indent=2))
        else:
            typer.echo(f"\n  Serving Analysis: {model} on {hardware}")
            typer.echo(f"  {'─' * 50}")
            typer.echo(f"  Sequence Length  : {seq_len}")
            typer.echo(f"  Batch Size       : {batch_size}")
            typer.echo(f"  Precision        : {precision}")
            typer.echo(f"  Efficiency       : {efficiency:.0%}")
            typer.echo(f"  Feasible         : {'Yes' if result.feasible else 'No'}")
            typer.echo(f"  {'─' * 50}")
            typer.echo(f"  TTFT (prefill)   : {result.ttft.to('ms'):.2f~P}")
            typer.echo(f"  ITL (decode)     : {result.itl.to('ms'):.4f~P}")
            typer.echo(f"  Tokens/s         : {tokens_per_sec:.1f}")
            typer.echo(f"  KV Cache         : {result.kv_cache_size.to('GB'):.2f~P}")
            typer.echo(f"  Memory Usage     : {result.memory_utilization:.1%}")
            typer.echo("")

    exit_with_code(ExitCode.SUCCESS)
