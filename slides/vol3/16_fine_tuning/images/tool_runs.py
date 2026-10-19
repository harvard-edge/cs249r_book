#!/usr/bin/env python3
"""Real tool runs behind four slides of Lecture 16 (Trajectory Fine-Tuning).

Each block runs a real library and writes its printed output to data/, so the
slide shows output produced by running the tool, not a mock-up.

  template: Qwen2.5-0.5B-Instruct's own chat template (tokenizer files from
            huggingface.co/Qwen/Qwen2.5-0.5B-Instruct, commit 7ae5576)
            renders a two-call tool turn; token counts by role.
  packing:  a randomly initialized 4-layer Llama-architecture model; logits of
            example 2 computed alone, packed under an ordinary causal mask,
            and packed under a block-diagonal mask with positions reset.
  accum:    transformers' ForCausalLMLoss on two micro-batches with different
            target counts: mean of micro-batch means against the global
            token average (the October 2024 gradient-accumulation fix).
  lora:     PEFT LoraConfig(r=16, q_proj and o_proj) on the Meta-Llama-3-8B
            architecture (config.json, huggingface.co/NousResearch/
            Meta-Llama-3-8B), built on the meta device: no weights loaded.

Run: <venv with torch, transformers, peft, numpy>/bin/python tool_runs.py TOKDIR
where TOKDIR holds the Qwen tokenizer files. Versions are printed in each file.
"""

import json
import sys
from pathlib import Path

import peft
import torch
import transformers
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from transformers.loss.loss_utils import ForCausalLMLoss

HERE = Path(__file__).parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)
VERS = (f"torch {torch.__version__}, transformers {transformers.__version__}, "
        f"peft {peft.__version__}")


def template(tokdir):
    tok = AutoTokenizer.from_pretrained(tokdir)
    tools = [
        {"type": "function", "function": {
            "name": "read_file", "description": "Read a file in the workspace.",
            "parameters": {"type": "object", "properties": {
                "target_uri": {"type": "string"}}, "required": ["target_uri"]}}},
        {"type": "function", "function": {
            "name": "run_tests", "description": "Run a test target.",
            "parameters": {"type": "object", "properties": {
                "target": {"type": "string"}}, "required": ["target"]}}},
    ]
    msgs = [
        {"role": "system", "content": "You repair code in an isolated working copy."},
        {"role": "user", "content": "Fix the connection-pool leak in src/pool.py"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "read_file",
             "arguments": {"target_uri": "src/pool.py"}}},
            {"type": "function", "function": {"name": "run_tests",
             "arguments": {"target": "tests/test_pool.py"}}}]},
        {"role": "tool", "content": "FAILED test_release: pool exhausted"},
        {"role": "tool", "content": "...file contents..."},
    ]
    text = tok.apply_chat_template(msgs, tools=tools, tokenize=False)
    # tokens by segment: render prefixes and difference their lengths
    def n(m):
        return len(tok(tok.apply_chat_template(m, tools=tools, tokenize=False))["input_ids"])
    n_sys_user, n_asst, n_all = n(msgs[:2]), n(msgs[:3]), n(msgs)
    out = DATA / "template_render.txt"
    tail = text[text.index("<|im_start|>user"):]
    out.write_text(
        f"# {VERS}; Qwen/Qwen2.5-0.5B-Instruct tokenizer @ 7ae5576\n"
        f"# apply_chat_template(msgs, tools=tools, tokenize=False), from the user turn on\n"
        + tail +
        f"\n# tokens: system+tools+user {n_sys_user}; assistant turn "
        f"{n_asst - n_sys_user}; two tool replies {n_all - n_asst}; total {n_all}\n")
    print(out.read_text())


def tiny_model():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=1000, hidden_size=128, intermediate_size=256,
                      num_hidden_layers=4, num_attention_heads=4,
                      num_key_value_heads=4, max_position_embeddings=512)
    m = LlamaForCausalLM(cfg).eval()
    m.config._attn_implementation = "eager"
    return m


def packing():
    m = tiny_model()
    g = torch.Generator().manual_seed(1)
    a = torch.randint(0, 1000, (1, 40), generator=g)   # example 1, 40 tokens
    b = torch.randint(0, 1000, (1, 24), generator=g)   # example 2, 24 tokens
    la, lb = a.shape[1], b.shape[1]
    packed = torch.cat([a, b], dim=1)
    L = la + lb
    with torch.no_grad():
        alone = m(b).logits[0]
        # ordinary causal mask, buffer-wide positions
        causal = m(packed).logits[0, la:]
        # block-diagonal mask (0 keep, -inf drop) and positions reset per example
        keep = torch.zeros(L, L, dtype=torch.bool)
        for s, e in ((0, la), (la, L)):
            keep[s:e, s:e] = torch.tril(torch.ones(e - s, e - s, dtype=torch.bool))
        mask = torch.zeros(1, 1, L, L)
        mask[0, 0][~keep] = torch.finfo(mask.dtype).min
        pos = torch.cat([torch.arange(la), torch.arange(lb)]).unsqueeze(0)
        block = m(packed, attention_mask=mask, position_ids=pos).logits[0, la:]
        # block mask but buffer-wide positions (RoPE: shift-invariant)
        block_nopos = m(packed, attention_mask=mask).logits[0, la:]
    d1 = (causal - alone).abs().max().item()
    d2 = (block - alone).abs().max().item()
    d3 = (block_nopos - alone).abs().max().item()
    out = DATA / "packing_test.txt"
    out.write_text(
        f"# {VERS}; random-init Llama architecture, 4 layers, fp32, eager attention\n"
        f"example 1: {la} tokens; example 2: {lb} tokens; one buffer of {L}\n"
        f"max |logit difference| of example 2, packed vs alone\n"
        f"  ordinary causal mask:            {d1:.3e}\n"
        f"  block-diagonal mask, pos reset:  {d2:.3e}\n"
        f"  block-diagonal mask, pos offset: {d3:.3e}\n")
    print(out.read_text())


def accum():
    torch.manual_seed(2)
    V = 1000
    # two micro-batches; labels -100 everywhere except the selected targets
    # micro-batch 1: a long trace whose targets the model already predicts well
    # (logits raised by 8 at the label); micro-batch 2: a short, unlearned one
    def micro(n_tok, n_tgt, boost):
        logits = torch.randn(1, n_tok, V)
        labels = torch.full((1, n_tok), -100)
        labels[0, n_tok - n_tgt:] = torch.randint(0, V, (n_tgt,))
        idx = torch.arange(n_tok - n_tgt - 1, n_tok - 1)
        logits[0, idx, labels[0, n_tok - n_tgt:]] += boost
        return logits, labels
    mb = [micro(4096, 2048, 8.0), micro(4096, 64, 0.0)]
    means = [ForCausalLMLoss(lg, lb, V).item() for lg, lb in mb]
    # count targets after the internal shift, as the Trainer does
    n_items = sum(int((lb[..., 1:] != -100).sum()) for _, lb in mb)
    summed = [ForCausalLMLoss(lg, lb, V, num_items_in_batch=torch.tensor(n_items)).item()
              for lg, lb in mb]
    tgt = [int((lb[..., 1:] != -100).sum()) for _, lb in mb]
    out = DATA / "accum_test.txt"
    out.write_text(
        f"# {VERS}; transformers.loss.loss_utils.ForCausalLMLoss, random logits; micro-batch 1 logits raised by 8 at its labels\n"
        f"micro-batch selected targets: {tgt[0]} and {tgt[1]}\n"
        f"micro-batch mean losses:       {means[0]:.4f}  {means[1]:.4f}\n"
        f"weight on each micro-batch:   mean of means 0.5000 0.5000; "
        f"global average {tgt[0]/n_items:.4f} {tgt[1]/n_items:.4f}\n"
        f"mean of micro-batch means:     {sum(means)/2:.4f}\n"
        f"with num_items_in_batch={n_items}: {sum(summed):.4f}\n")
    print(out.read_text())


def lora():
    cfg = json.loads((DATA / "llama3_8b_config.json").read_text())
    lc = LlamaConfig(**{k: v for k, v in cfg.items()
                        if k not in ("architectures", "transformers_version", "torch_dtype")})
    with torch.device("meta"):
        model = LlamaForCausalLM(lc)
    pcfg = peft.LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "o_proj"],
                           lora_dropout=0.0, task_type="CAUSAL_LM")
    pm = peft.get_peft_model(model, pcfg)
    trainable, total = pm.get_nb_trainable_parameters()
    out = DATA / "lora_count.txt"
    out.write_text(
        f"# {VERS}; Meta-Llama-3-8B config.json, meta device (no weights)\n"
        f"LoraConfig(r=16, lora_alpha=32, target_modules=['q_proj', 'o_proj'])\n"
        f"trainable params: {trainable:,} || all params: {total:,} || "
        f"trainable%: {100*trainable/total:.4f}\n")
    print(out.read_text())
    # the adapter's own release record, as PEFT writes it
    import tempfile
    pcfg.base_model_name_or_path = "meta-llama/Meta-Llama-3-8B"
    with tempfile.TemporaryDirectory() as d:
        pcfg.save_pretrained(d)
        (DATA / "adapter_config.json").write_text(Path(d, "adapter_config.json").read_text())


if __name__ == "__main__":
    template(sys.argv[1])
    packing()
    accum()
    lora()
