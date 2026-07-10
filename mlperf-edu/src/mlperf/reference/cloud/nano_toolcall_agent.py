"""
MLPerf EDU: Nano-ToolCall Agent Benchmark

A pedagogical structured-output generation pipeline that measures the systems
cost of constrained decoding: LLM → JSON tool call → parse → dispatch → return.

Architecture:
    User Query → Transformer generates structured JSON output
    → JSON parse + validate schema → Dispatch to correct function
    → Return result

Systems Focus:
    - Structured output generation vs free-form generation throughput
    - JSON parse validation overhead
    - Function dispatch latency
    - Students compare constrained vs unconstrained decoding performance

Quality Target:
    - Training: Cross-entropy loss on structured output token prediction
    - Inference: Queries/second, JSON validity rate, correct function selection rate

Provenance: OpenAI function calling paradigm — measures systems cost
of structured output generation for tool-augmented LLMs
"""

import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Function Registry — mock API functions the agent can call
# ---------------------------------------------------------------------------

class FunctionRegistry:
    """
    A registry of mock API functions that the tool-calling agent can invoke.

    Each function has a name, description, parameter schema, and implementation.
    The agent must generate a JSON object selecting the correct function and
    providing valid arguments.
    """

    FUNCTIONS = {
        "get_weather": {
            "description": "Get current weather for a city",
            "parameters": {"city": "str"},
            "implementation": lambda args: f"72°F, sunny in {args.get('city', 'unknown')}",
        },
        "calculate": {
            "description": "Perform arithmetic calculation",
            "parameters": {"expression": "str"},
            "implementation": lambda args: str(eval(
                args.get("expression", "0"),
                {"__builtins__": {"abs": abs, "round": round, "min": min, "max": max}}
            )),
        },
        "search": {
            "description": "Search a knowledge base for information",
            "parameters": {"query": "str"},
            "implementation": lambda args: f"Found 3 results for '{args.get('query', '')}'",
        },
        "translate": {
            "description": "Translate text to another language",
            "parameters": {"text": "str", "target_lang": "str"},
            "implementation": lambda args: f"[{args.get('target_lang', 'en')}] {args.get('text', '')}",
        },
        "summarize": {
            "description": "Summarize a text passage",
            "parameters": {"text": "str", "max_words": "int"},
            "implementation": lambda args: " ".join(
                args.get("text", "").split()[:args.get("max_words", 10)]
            ),
        },
        "set_reminder": {
            "description": "Set a reminder for a specific time",
            "parameters": {"message": "str", "time": "str"},
            "implementation": lambda args: f"Reminder set: '{args.get('message', '')}' at {args.get('time', 'now')}",
        },
        "convert_units": {
            "description": "Convert between measurement units",
            "parameters": {"value": "float", "from_unit": "str", "to_unit": "str"},
            "implementation": lambda args: f"{args.get('value', 0)} {args.get('from_unit', '')} = {float(args.get('value', 0)) * 1.0} {args.get('to_unit', '')}",
        },
        "format_date": {
            "description": "Format a date string",
            "parameters": {"date": "str", "format": "str"},
            "implementation": lambda args: f"Formatted: {args.get('date', 'today')} as {args.get('format', 'ISO')}",
        },
        "count_words": {
            "description": "Count words in a text",
            "parameters": {"text": "str"},
            "implementation": lambda args: str(len(args.get("text", "").split())),
        },
        "validate_email": {
            "description": "Check if an email address is valid",
            "parameters": {"email": "str"},
            "implementation": lambda args: str("@" in args.get("email", "") and "." in args.get("email", "")),
        },
    }

    # Query → expected function mapping for evaluation
    QUERY_BANK = [
        ("What's the weather in Boston?", "get_weather", {"city": "Boston"}),
        ("Calculate 15 * 8 + 2", "calculate", {"expression": "15 * 8 + 2"}),
        ("Search for machine learning papers", "search", {"query": "machine learning papers"}),
        ("Translate 'hello world' to Spanish", "translate", {"text": "hello world", "target_lang": "es"}),
        ("How many words in 'the quick brown fox'?", "count_words", {"text": "the quick brown fox"}),
        ("Is test@example.com a valid email?", "validate_email", {"email": "test@example.com"}),
        ("Set a reminder to submit homework at 5pm", "set_reminder", {"message": "submit homework", "time": "5pm"}),
        ("Convert 100 from celsius to fahrenheit", "convert_units", {"value": 100, "from_unit": "celsius", "to_unit": "fahrenheit"}),
    ]

    @staticmethod
    def validate_call(call_json: dict) -> tuple[bool, str]:
        """
        Validate a function call JSON object.

        Expected format: {"function": "name", "arguments": {...}}

        Returns (is_valid, error_or_result)
        """
        if not isinstance(call_json, dict):
            return False, "Call must be a JSON object"

        func_name = call_json.get("function")
        if func_name not in FunctionRegistry.FUNCTIONS:
            return False, f"Unknown function: {func_name}"

        arguments = call_json.get("arguments", {})
        if not isinstance(arguments, dict):
            return False, "Arguments must be a JSON object"

        # Check required parameters
        schema = FunctionRegistry.FUNCTIONS[func_name]["parameters"]
        for param_name in schema:
            if param_name not in arguments:
                return False, f"Missing parameter: {param_name}"

        return True, "valid"

    @staticmethod
    def execute_call(call_json: dict) -> tuple[bool, str]:
        """Validate and execute a function call."""
        valid, msg = FunctionRegistry.validate_call(call_json)
        if not valid:
            return False, msg

        func_name = call_json["function"]
        arguments = call_json["arguments"]

        try:
            impl = FunctionRegistry.FUNCTIONS[func_name]["implementation"]
            result = impl(arguments)
            return True, result
        except Exception as e:
            return False, f"Execution error: {e}"

    @staticmethod
    def list_functions() -> list[str]:
        return list(FunctionRegistry.FUNCTIONS.keys())


# ---------------------------------------------------------------------------
# Nano-ToolCall Agent Model
# ---------------------------------------------------------------------------

class NanoToolCallAgent(nn.Module):
    """
    A transformer that generates structured function-call outputs.

    Two output heads:
    1. lm_head: standard next-token prediction (used during training)
    2. function_head: classifies which function to call (10 functions)

    The model learns to map query tokens → function selection, which is the
    core structured-output pattern in modern tool-using LLMs.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        max_seq_len: int = 128,
        n_functions: int = 10,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.n_functions = n_functions

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

        # Dual heads
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.function_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_functions),
        )

    def forward(self, input_ids: torch.Tensor, targets=None):
        """
        Standard forward pass for training.

        Returns:
            logits: (B, T, vocab_size)
            loss: scalar if targets provided
        """
        B, T = input_ids.size()
        T = min(T, self.max_seq_len)
        input_ids = input_ids[:, :T]

        pos = torch.arange(0, T, device=input_ids.device)
        x = self.token_embed(input_ids) + self.pos_embed(pos)

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

    def predict_function(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Predict which function to call given input tokens.

        Returns:
            function_logits: (B, n_functions)
        """
        B, T = input_ids.size()
        T = min(T, self.max_seq_len)
        input_ids = input_ids[:, :T]

        pos = torch.arange(0, T, device=input_ids.device)
        x = self.token_embed(input_ids) + self.pos_embed(pos)

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
        # Pool sequence → function classification
        pooled = hidden.mean(dim=1)
        return self.function_head(pooled)

    def forward_with_timing(
        self, input_ids: torch.Tensor, n_queries: int = 10
    ):
        """
        Benchmark the full function-calling pipeline over multiple queries.

        For each query:
        1. Generate tokens (measure generation latency)
        2. Predict function (measure classification latency)
        3. Validate + execute function call (measure dispatch latency)

        Returns timing breakdown.
        """
        self.eval()
        func_names = FunctionRegistry.list_functions()

        results = {
            "queries": [],
            "total_generation_ms": 0.0,
            "total_classification_ms": 0.0,
            "total_dispatch_ms": 0.0,
            "total_ms": 0.0,
            "valid_calls": 0,
            "total_queries": n_queries,
        }

        with torch.no_grad():
            for i in range(n_queries):
                query_result = {"query_idx": i}

                # Phase 1: Generate tokens (structured output)
                t0 = time.perf_counter()
                logits, _ = self.forward(input_ids)
                gen_ms = (time.perf_counter() - t0) * 1000
                query_result["generation_ms"] = gen_ms
                results["total_generation_ms"] += gen_ms

                # Phase 2: Classify which function to call
                t0 = time.perf_counter()
                func_logits = self.predict_function(input_ids)
                func_idx = func_logits.argmax(dim=1)[0].item()
                selected_func = func_names[func_idx % len(func_names)]
                cls_ms = (time.perf_counter() - t0) * 1000
                query_result["classification_ms"] = cls_ms
                query_result["selected_function"] = selected_func
                results["total_classification_ms"] += cls_ms

                # Phase 3: Build and execute function call
                t0 = time.perf_counter()
                call = {
                    "function": selected_func,
                    "arguments": {
                        k: "test" for k in
                        FunctionRegistry.FUNCTIONS[selected_func]["parameters"]
                    },
                }
                valid, output = FunctionRegistry.execute_call(call)
                dispatch_ms = (time.perf_counter() - t0) * 1000
                query_result["dispatch_ms"] = dispatch_ms
                query_result["call_valid"] = valid
                results["total_dispatch_ms"] += dispatch_ms

                if valid:
                    results["valid_calls"] += 1

                results["queries"].append(query_result)

        results["total_ms"] = (
            results["total_generation_ms"]
            + results["total_classification_ms"]
            + results["total_dispatch_ms"]
        )
        results["queries_per_second"] = (
            n_queries / (results["total_ms"] / 1000) if results["total_ms"] > 0 else 0
        )

        return results


if __name__ == "__main__":
    print("🚀 Nano-ToolCall Agent Benchmark — Architecture Demo")

    model = NanoToolCallAgent()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Parameters: ~{total_params/1e6:.1f}M")

    # Training demo
    dummy_input = torch.randint(0, 50257, (2, 32))
    dummy_target = torch.randint(0, 50257, (2, 32))
    logits, loss = model(dummy_input, targets=dummy_target)
    print(f"✅ Training forward: logits={logits.shape}, loss={loss.item():.4f}")

    # Function prediction demo
    func_logits = model.predict_function(dummy_input)
    func_names = FunctionRegistry.list_functions()
    for b in range(min(2, dummy_input.size(0))):
        idx = func_logits[b].argmax().item()
        print(f"✅ Batch {b}: predicted function = {func_names[idx % len(func_names)]}")

    # Full pipeline timing
    results = model.forward_with_timing(dummy_input, n_queries=5)
    print(f"\n✅ ToolCall pipeline ({results['total_queries']} queries):")
    print(f"   Generation:      {results['total_generation_ms']:.1f}ms")
    print(f"   Classification:  {results['total_classification_ms']:.1f}ms")
    print(f"   Dispatch:        {results['total_dispatch_ms']:.2f}ms")
    print(f"   Valid calls:     {results['valid_calls']}/{results['total_queries']}")
    print(f"   QPS:             {results['queries_per_second']:.1f}")

    # Function registry demo
    print(f"\n🔧 Registered functions: {FunctionRegistry.list_functions()}")
    test_call = {"function": "calculate", "arguments": {"expression": "2 + 3"}}
    ok, result = FunctionRegistry.execute_call(test_call)
    print(f"   calculate('2 + 3') → {result}")

    bad_call = {"function": "nonexistent", "arguments": {}}
    ok, result = FunctionRegistry.execute_call(bad_call)
    print(f"   nonexistent() → OK={ok}, {result}")
