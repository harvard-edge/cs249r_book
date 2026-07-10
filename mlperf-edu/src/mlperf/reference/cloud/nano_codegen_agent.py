"""
MLPerf EDU: Nano-CodeGen Agent Benchmark

A pedagogical code-generation-and-verification pipeline that exposes the
systems cost of iterative LLM → execute → verify → retry loops.

Architecture:
    Task Prompt → Transformer generates token sequence → AST parse check
    → Sandboxed execution → Output verification → If wrong, retry with
    error feedback appended to prompt (growing context window)

Systems Focus:
    - Each retry iteration grows the prompt (linear context = quadratic attention)
    - Students measure tokens-per-attempt, wall-clock per iteration, memory growth
    - The retry loop is the canonical agentic pattern: observe → reason → act → observe

Quality Target:
    - Training: Cross-entropy loss on code token prediction
    - Inference: Iterations-to-correct, total tokens generated, wall-clock time
"""

import ast
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class CodeVerifier:
    """
    Sandboxed code execution and verification engine.

    Evaluates generated code by:
    1. AST parsing (syntax check)
    2. Sandboxed execution (restricted builtins)
    3. Output comparison against expected result

    This is a pedagogical stand-in for the kind of verification that
    SWE-bench or HumanEval would do. The focus here is on measuring
    the systems overhead of the verify step, not the correctness of
    a real LLM's code output.
    """

    # Simple tasks for the benchmark: (description, expected_output)
    TASK_BANK = [
        ("Return the sum of [1, 2, 3, 4, 5]", "15"),
        ("Return 'hello' reversed", "olleh"),
        ("Return the length of 'benchmark'", "9"),
        ("Return 2 ** 10", "1024"),
        ("Return sorted([3, 1, 4, 1, 5])", "[1, 1, 3, 4, 5]"),
        ("Return max(10, 20, 5)", "20"),
        ("Return 'ab' * 3", "ababab"),
        ("Return list(range(5))", "[0, 1, 2, 3, 4]"),
    ]

    @staticmethod
    def check_syntax(code_str: str) -> tuple[bool, str]:
        """Validate Python syntax via AST parsing."""
        try:
            ast.parse(code_str)
            return True, "OK"
        except SyntaxError as e:
            return False, f"SyntaxError: {e}"

    @staticmethod
    def check_safety(code_str: str) -> tuple[bool, str]:
        """Check for forbidden constructs (imports, exec, eval, open)."""
        try:
            tree = ast.parse(code_str)
        except SyntaxError:
            return False, "Cannot parse"

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                return False, "Forbidden: import statement"
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in ("exec", "eval", "open", "__import__"):
                    return False, f"Forbidden: {func.id}()"
        return True, "OK"

    @staticmethod
    def execute(code_str: str, timeout_ms: int = 100) -> tuple[bool, str]:
        """
        Execute code in a restricted sandbox and capture the result.
        The code should end with a bare expression whose value is the 'result'.
        """
        safe, reason = CodeVerifier.check_safety(code_str)
        if not safe:
            return False, reason

        try:
            # Restricted globals — no file I/O, no imports
            restricted_globals = {
                "__builtins__": {
                    "range": range, "len": len, "sum": sum, "max": max, "min": min,
                    "sorted": sorted, "list": list, "str": str, "int": int,
                    "float": float, "abs": abs, "round": round, "enumerate": enumerate,
                    "zip": zip, "map": map, "filter": filter, "reversed": reversed,
                    "True": True, "False": False, "None": None,
                }
            }
            local_vars = {}
            exec(code_str, restricted_globals, local_vars)
            result = local_vars.get("result", None)
            return True, str(result)
        except Exception as e:
            return False, f"RuntimeError: {e}"


class NanoCodeGenAgent(nn.Module):
    """
    The CodeGen agent model.

    This is a small autoregressive transformer trained on synthetic code tokens.
    In the benchmark loop, it generates code → the CodeVerifier checks it →
    if wrong, the error is appended to the prompt and the model retries.

    The key systems metric is: how does inference cost grow with each retry
    iteration as the prompt context expands?
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        max_seq_len: int = 256,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        self.layers = nn.ModuleList([
            nn.ModuleDict(dict(
                ln_1=nn.LayerNorm(d_model),
                attn=nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                ln_2=nn.LayerNorm(d_model),
                ffn=nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Linear(d_model * 4, d_model),
                ),
            ))
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Feedback projection: encodes error messages back into the model's space
        # In a real agent, this would be tokenized text. Here we use a learned
        # embedding for each retry iteration to simulate growing context.
        self.feedback_embed = nn.Embedding(16, d_model)  # up to 16 retries

    def forward(self, input_ids: torch.Tensor, targets=None, retry_step: int = 0):
        """
        Forward pass with optional retry-step conditioning.

        Args:
            input_ids: (B, T) token IDs
            targets: (B, T) target token IDs for training
            retry_step: current retry iteration (0 = first attempt)

        Returns:
            logits: (B, T, vocab_size)
            loss: scalar if targets provided
        """
        B, T = input_ids.size()
        T = min(T, self.max_seq_len)
        input_ids = input_ids[:, :T]

        pos = torch.arange(0, T, device=input_ids.device)
        x = self.token_embed(input_ids) + self.pos_embed(pos)

        # Inject retry-step conditioning
        if retry_step > 0:
            step_idx = torch.tensor(
                min(retry_step, 15), device=input_ids.device
            )
            feedback_signal = self.feedback_embed(step_idx)
            x = x + feedback_signal.unsqueeze(0).unsqueeze(0)

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )

        for block in self.layers:
            attn_out, _ = block["attn"](
                block["ln_1"](x), block["ln_1"](x), block["ln_1"](x),
                attn_mask=causal_mask, need_weights=False
            )
            x = x + attn_out
            x = x + block["ffn"](block["ln_2"](x))

        logits = self.lm_head(self.ln_f(x))

        loss = None
        if targets is not None:
            targets = targets[:, :T]
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size), targets.view(-1)
            )

        return logits, loss

    def forward_with_timing(
        self, input_ids: torch.Tensor, max_retries: int = 5
    ):
        """
        Simulates the agentic retry loop with per-iteration timing.

        This measures the key systems cost: each retry grows the effective
        context, and attention cost scales quadratically.

        Returns:
            results: dict with per-iteration timings and total metrics
        """
        self.eval()
        results = {
            "iterations": [],
            "total_tokens_generated": 0,
            "total_ms": 0.0,
        }

        with torch.no_grad():
            for attempt in range(max_retries):
                t0 = time.perf_counter()

                # Simulate growing context: each retry adds more tokens
                # (in a real agent, this would be error feedback appended)
                ctx_growth = attempt * 16
                if ctx_growth > 0:
                    extra = torch.randint(
                        0, self.vocab_size,
                        (input_ids.size(0), ctx_growth),
                        device=input_ids.device
                    )
                    grown_input = torch.cat([input_ids, extra], dim=1)
                else:
                    grown_input = input_ids

                logits, _ = self.forward(grown_input, retry_step=attempt)
                elapsed_ms = (time.perf_counter() - t0) * 1000

                tokens_this_iter = grown_input.size(1)
                results["iterations"].append({
                    "attempt": attempt,
                    "context_length": tokens_this_iter,
                    "latency_ms": elapsed_ms,
                })
                results["total_tokens_generated"] += tokens_this_iter
                results["total_ms"] += elapsed_ms

        return results


if __name__ == "__main__":
    print("🚀 Nano-CodeGen Agent Benchmark — Architecture Demo")

    model = NanoCodeGenAgent(
        vocab_size=50257, d_model=128, n_heads=4, n_layers=4
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Parameters: ~{total_params/1e6:.1f}M")

    # Training mode demo
    dummy_input = torch.randint(0, 50257, (4, 64))
    dummy_target = torch.randint(0, 50257, (4, 64))
    logits, loss = model(dummy_input, targets=dummy_target)
    print(f"✅ Training forward pass: logits={logits.shape}, loss={loss.item():.4f}")

    # Agentic retry loop timing demo
    results = model.forward_with_timing(dummy_input, max_retries=5)
    print(f"✅ Agentic retry loop ({len(results['iterations'])} iterations):")
    for it in results["iterations"]:
        print(f"   Attempt {it['attempt']}: ctx_len={it['context_length']}, "
              f"latency={it['latency_ms']:.2f} ms")
    print(f"   Total: {results['total_tokens_generated']} tokens, "
          f"{results['total_ms']:.2f} ms")

    # Code verifier demo
    print("\n🔍 CodeVerifier Demo:")
    verifier = CodeVerifier()
    test_code = "result = sum([1, 2, 3, 4, 5])"
    ok, output = verifier.execute(test_code)
    print(f"   Code: '{test_code}' → OK={ok}, output='{output}'")

    bad_code = "import os; result = os.getcwd()"
    ok, output = verifier.execute(bad_code)
    print(f"   Code: '{bad_code}' → OK={ok}, output='{output}'")
