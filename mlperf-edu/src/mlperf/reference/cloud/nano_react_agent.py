"""
MLPerf EDU: Nano-ReAct Agent Benchmark

A pedagogical Reasoning + Acting loop that exposes the systems cost of
multi-step tool-augmented inference.

Architecture:
    Question → Think (transformer generates reasoning tokens)
    → Act (select + invoke tool from registry)
    → Observe (parse tool output, append to context)
    → Repeat until final answer or max steps

Systems Focus:
    - KV-cache growth per reasoning step (memory scaling)
    - Tool dispatch latency vs. generation latency
    - Total wall-clock for multi-step reasoning chains
    - Students measure how each additional step degrades throughput

Quality Target:
    - Training: Cross-entropy loss on reasoning trace prediction
    - Inference: Steps-to-answer, total reasoning time, memory per step

Provenance: Yao et al. 2023, "ReAct: Synergizing Reasoning and Acting in Language Models"
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# Tool Registry — deterministic tools students can profile
# ---------------------------------------------------------------------------

class ToolRegistry:
    """
    A bank of simple, deterministic tools that a ReAct agent can invoke.

    Each tool is a pure function: (str) -> str. Students measure the dispatch
    overhead and can compare tool execution time vs. LLM reasoning time.
    """

    TOOLS = {
        "calculator": {
            "description": "Evaluate a simple arithmetic expression",
            "examples": ["calculator(2 + 3)", "calculator(10 * 5)"],
        },
        "string_length": {
            "description": "Return the length of a string",
            "examples": ["string_length('hello')", "string_length('benchmark')"],
        },
        "lookup": {
            "description": "Look up a value in a key-value store",
            "examples": ["lookup('pi')", "lookup('e')"],
        },
        "compare": {
            "description": "Compare two numbers, return 'greater', 'less', or 'equal'",
            "examples": ["compare(5, 3)", "compare(2, 2)"],
        },
    }

    # Simple lookup table for the lookup tool
    LOOKUP_TABLE = {
        "pi": "3.14159",
        "e": "2.71828",
        "sqrt2": "1.41421",
        "golden_ratio": "1.61803",
        "avogadro": "6.022e23",
        "speed_of_light": "299792458",
        "planck": "6.626e-34",
        "boltzmann": "1.381e-23",
    }

    @staticmethod
    def execute(tool_name: str, argument: str) -> tuple[bool, str]:
        """
        Execute a tool call and return (success, result).

        Args:
            tool_name: one of the registered tool names
            argument: string argument to the tool

        Returns:
            (True, result_string) on success
            (False, error_message) on failure
        """
        try:
            if tool_name == "calculator":
                # Restricted eval: only digits, operators, parentheses, decimals
                allowed = set("0123456789+-*/().% ")
                if not all(c in allowed for c in argument):
                    return False, f"Invalid characters in expression: {argument}"
                result = eval(argument, {"__builtins__": {}})
                return True, str(result)

            elif tool_name == "string_length":
                # Strip quotes if present
                s = argument.strip("'\"")
                return True, str(len(s))

            elif tool_name == "lookup":
                key = argument.strip("'\"").lower()
                value = ToolRegistry.LOOKUP_TABLE.get(key)
                if value is None:
                    return False, f"Key '{key}' not found in lookup table"
                return True, value

            elif tool_name == "compare":
                parts = argument.split(",")
                if len(parts) != 2:
                    return False, "compare requires exactly 2 comma-separated numbers"
                a, b = float(parts[0].strip()), float(parts[1].strip())
                if a > b:
                    return True, "greater"
                elif a < b:
                    return True, "less"
                else:
                    return True, "equal"

            else:
                return False, f"Unknown tool: {tool_name}"

        except Exception as e:
            return False, f"Tool execution error: {e}"

    @staticmethod
    def list_tools() -> list[str]:
        return list(ToolRegistry.TOOLS.keys())


# ---------------------------------------------------------------------------
# ReAct Reasoning Task Bank
# ---------------------------------------------------------------------------

class ReActTaskBank:
    """
    Multi-step problems that require 2-5 tool invocations to solve.

    Each task specifies the expected tool call sequence and final answer,
    allowing the benchmark to measure both correctness and systems cost.
    """

    TASKS = [
        {
            "question": "What is (25 * 4) + (10 * 3)?",
            "expected_steps": [
                ("calculator", "25 * 4"),
                ("calculator", "10 * 3"),
                ("calculator", "100 + 30"),
            ],
            "expected_answer": "130",
        },
        {
            "question": "Is the length of 'benchmark' greater than 5?",
            "expected_steps": [
                ("string_length", "benchmark"),
                ("compare", "9, 5"),
            ],
            "expected_answer": "greater",
        },
        {
            "question": "What is pi times 2?",
            "expected_steps": [
                ("lookup", "pi"),
                ("calculator", "3.14159 * 2"),
            ],
            "expected_answer": "6.28318",
        },
        {
            "question": "Which is larger: the length of 'hello' or the length of 'world!'?",
            "expected_steps": [
                ("string_length", "hello"),
                ("string_length", "world!"),
                ("compare", "5, 6"),
            ],
            "expected_answer": "less",
        },
    ]


# ---------------------------------------------------------------------------
# Nano-ReAct Agent Model
# ---------------------------------------------------------------------------

class NanoReActAgent(nn.Module):
    """
    A small transformer that generates reasoning tokens and tool-selection logits.

    The model has two output heads:
    1. lm_head: standard next-token prediction (for reasoning traces)
    2. tool_head: classifies which tool to invoke (for action steps)

    The key insight: at each reasoning step, the full context must be
    re-processed (or KV-cached), and the context grows with each
    Think → Act → Observe cycle.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        max_seq_len: int = 256,
        n_tools: int = 4,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.n_tools = n_tools

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        # Step embedding: encodes which reasoning step we're on (0-15)
        self.step_embed = nn.Embedding(16, d_model)

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

        # Dual heads
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.tool_head = nn.Linear(d_model, n_tools)

    def forward(self, input_ids: torch.Tensor, targets=None, step: int = 0):
        """
        Forward pass with step conditioning.

        Args:
            input_ids: (B, T) token IDs
            targets: (B, T) for training loss
            step: current reasoning step (0-indexed)

        Returns:
            logits: (B, T, vocab_size) next-token prediction
            loss: scalar if targets provided
        """
        B, T = input_ids.size()
        T = min(T, self.max_seq_len)
        input_ids = input_ids[:, :T]

        pos = torch.arange(0, T, device=input_ids.device)
        x = self.token_embed(input_ids) + self.pos_embed(pos)

        # Inject step conditioning
        step_idx = torch.tensor(min(step, 15), device=input_ids.device)
        x = x + self.step_embed(step_idx)

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

        hidden = self.ln_f(x)
        logits = self.lm_head(hidden)

        loss = None
        if targets is not None:
            targets = targets[:, :T]
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size), targets.reshape(-1)
            )

        return logits, loss

    def predict_tool(self, input_ids: torch.Tensor, step: int = 0):
        """
        Predict which tool to invoke from the tool registry.

        Returns:
            tool_logits: (B, n_tools) tool selection probabilities
        """
        B, T = input_ids.size()
        T = min(T, self.max_seq_len)
        input_ids = input_ids[:, :T]

        pos = torch.arange(0, T, device=input_ids.device)
        x = self.token_embed(input_ids) + self.pos_embed(pos)

        step_idx = torch.tensor(min(step, 15), device=input_ids.device)
        x = x + self.step_embed(step_idx)

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

        hidden = self.ln_f(x)
        # Pool over sequence for tool classification
        pooled = hidden.mean(dim=1)
        tool_logits = self.tool_head(pooled)
        return tool_logits

    def forward_with_timing(
        self, input_ids: torch.Tensor, max_steps: int = 5
    ):
        """
        Simulate the full ReAct loop with per-step timing.

        Measures:
        - Reasoning latency per step (transformer forward)
        - Tool dispatch latency per step
        - Context growth per step
        - Total wall-clock time

        Returns dict with per-step metrics.
        """
        self.eval()
        tool_names = ToolRegistry.list_tools()
        results = {
            "steps": [],
            "total_reasoning_ms": 0.0,
            "total_tool_ms": 0.0,
            "total_ms": 0.0,
        }

        current_context = input_ids

        def _get_memory_bytes():
            """Get current memory usage (platform-aware)."""
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated()
            elif hasattr(torch.mps, 'current_allocated_memory'):
                try:
                    return torch.mps.current_allocated_memory()
                except Exception:
                    pass
            # Fallback: estimate from context tensor size
            return current_context.nelement() * current_context.element_size()

        with torch.no_grad():
            for step in range(max_steps):
                step_result = {
                    "step": step,
                    "context_length": current_context.size(1),
                    "memory_bytes": _get_memory_bytes(),
                }

                # Phase 1: Reason (transformer forward pass)
                t0 = time.perf_counter()
                logits, _ = self.forward(current_context, step=step)
                tool_logits = self.predict_tool(current_context, step=step)
                reasoning_ms = (time.perf_counter() - t0) * 1000
                step_result["reasoning_ms"] = reasoning_ms
                results["total_reasoning_ms"] += reasoning_ms

                # Phase 2: Act (select and invoke tool)
                t0 = time.perf_counter()
                tool_idx = tool_logits.argmax(dim=1)[0].item()
                selected_tool = tool_names[tool_idx % len(tool_names)]
                # Use a dummy argument for benchmarking
                ok, output = ToolRegistry.execute(selected_tool, "42")
                tool_ms = (time.perf_counter() - t0) * 1000
                step_result["tool_ms"] = tool_ms
                step_result["tool_selected"] = selected_tool
                results["total_tool_ms"] += tool_ms

                # Phase 3: Observe (grow context with observation tokens)
                # This simulates the KV-cache growth in production ReAct agents:
                # each Think→Act→Observe cycle appends tokens to the context,
                # causing quadratic attention cost growth.
                observation_tokens = torch.randint(
                    0, self.vocab_size,
                    (current_context.size(0), 8),
                    device=current_context.device
                )
                current_context = torch.cat(
                    [current_context, observation_tokens], dim=1
                )

                results["steps"].append(step_result)

        results["total_ms"] = results["total_reasoning_ms"] + results["total_tool_ms"]
        results["final_context_length"] = current_context.size(1)
        results["final_memory_bytes"] = _get_memory_bytes()
        return results


if __name__ == "__main__":
    print("🚀 Nano-ReAct Agent Benchmark — Architecture Demo")

    model = NanoReActAgent()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Parameters: ~{total_params/1e6:.1f}M")

    # Training demo
    dummy_input = torch.randint(0, 50257, (2, 32))
    dummy_target = torch.randint(0, 50257, (2, 32))
    logits, loss = model(dummy_input, targets=dummy_target)
    print(f"✅ Training forward: logits={logits.shape}, loss={loss.item():.4f}")

    # Tool prediction demo
    tool_logits = model.predict_tool(dummy_input)
    print(f"✅ Tool prediction: {tool_logits.shape} → selected tool idx={tool_logits.argmax(1).tolist()}")

    # Full ReAct loop timing
    results = model.forward_with_timing(dummy_input, max_steps=4)
    print(f"✅ ReAct loop ({len(results['steps'])} steps):")
    for s in results["steps"]:
        print(f"   Step {s['step']}: ctx={s['context_length']}, "
              f"reason={s['reasoning_ms']:.1f}ms, tool={s['tool_ms']:.2f}ms "
              f"[{s['tool_selected']}]")
    print(f"   Final context: {results['final_context_length']} tokens")
    print(f"   Total: reasoning={results['total_reasoning_ms']:.1f}ms, "
          f"tools={results['total_tool_ms']:.2f}ms")

    # Tool registry demo
    print(f"\n🔧 Tool Registry: {ToolRegistry.list_tools()}")
    for name in ToolRegistry.list_tools():
        ok, out = ToolRegistry.execute(name, "42")
        print(f"   {name}('42') → OK={ok}, result={out}")
