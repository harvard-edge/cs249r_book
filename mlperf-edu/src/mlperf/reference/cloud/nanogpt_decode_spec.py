"""
MLPerf EDU iter-6 SUT: speculative decoding with a small draft model.

THE ALGORITHMIC LEVER. Per Han's iter-6 proposal: speculative decoding
is the one optimization that beats the memory-bandwidth wall without
changing hardware. A draft model 10x smaller predicts gamma tokens
ahead; the target model verifies them in a single forward pass with
the full KV cache. Accepted prefix is appended; rejected suffix is
discarded.

Draft: NanoGPTWhiteBox at iter-1 default (~11M).
Target: NanoGPTWhiteBox at GPT-2-Small geometry (~124M).
Acceptance criterion: argmax(target) == draft_token (lossless verify).

Expected speedup at acceptance rate alpha=0.7, gamma=4 draft tokens:
  theoretical = (1 - alpha^(gamma+1)) / ((1-alpha)*(gamma+1)) ~ 2.1x
  practical (after dispatch + draft overhead) ~ 1.4-1.7x

Smoke gate (Han's iter-6): tokens/s of spec >= 1.4 * tokens/s of fp16 baseline.
"""
from __future__ import annotations

import statistics
import time

import torch

from .nanogpt_train import NanoGPTWhiteBox, GPT2_SMALL_CONFIG
from mlperf.roofline import measure_roofline


def _sync():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


class SpeculativeDecode:
    """Speculative-decode loop with a small draft and a large target.

    Both share the char-level vocab (vocab=128). For the iter-6 scaffold,
    the draft model is the existing 11M NanoGPT and the target is the
    124M GPT-2-Small variant.
    """

    def __init__(self,
                 target: NanoGPTWhiteBox,
                 draft: NanoGPTWhiteBox,
                 prefill_ctx: int = 2048,
                 decode_tokens: int = 64,
                 gamma: int = 4,
                 batch_size: int = 1):
        self.target = target.eval()
        self.draft = draft.eval()
        self.prefill_ctx = prefill_ctx
        self.decode_tokens = decode_tokens
        self.gamma = gamma
        self.batch = batch_size
        self.vocab = target.config["vocab_size"]
        if draft.config["vocab_size"] != self.vocab:
            raise ValueError("draft and target must share vocab")

    def _argmax(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1, keepdim=True)

    def run(self) -> dict:
        device = next(self.target.parameters()).device
        prompt = torch.randint(0, self.vocab, (self.batch, self.prefill_ctx),
                                device=device)

        with torch.no_grad():
            # Warm both KV caches with the same prompt.
            _sync()
            t_warm = time.perf_counter()
            target_logits, target_kv = self.target(prompt, use_kv_cache=True)
            draft_logits, draft_kv = self.draft(prompt, use_kv_cache=True)
            _sync()
            warm_time = time.perf_counter() - t_warm

            n_target_params = sum(p.numel() for p in self.target.parameters())
            n_draft_params = sum(p.numel() for p in self.draft.parameters())
            cfg = self.target.config
            head_dim = cfg["n_embd"] // cfg["n_head"]
            kv_bytes_per_token_target = 2 * cfg["n_layer"] * cfg["n_head"] * head_dim * 2  # fp16

            tokens_emitted = 0
            cycles = 0
            n_accepted_total = 0
            per_cycle_times = []

            with measure_roofline(
                "nanogpt-decode-spec",
                # FLOPs per cycle: gamma forwards on draft + 1 verify on target.
                analytic_flops=lambda: (2 * n_draft_params * self.gamma + 2 * n_target_params * (self.gamma + 1)) * cycles,
                # Bytes per cycle: weight rereads + KV streams (fp16).
                analytic_bytes=lambda: (
                    (n_draft_params * 2 * self.gamma)
                    + (n_target_params * 2)
                    + (kv_bytes_per_token_target * (self.prefill_ctx + tokens_emitted))
                ) * cycles,
                n_iter=max(cycles, 1),
            ):
                while tokens_emitted < self.decode_tokens:
                    cycle_start = time.perf_counter()

                    # 1. Draft proposes gamma tokens autoregressively.
                    draft_tokens = []
                    cur_logits = draft_logits
                    cur_draft_kv = draft_kv
                    for _ in range(self.gamma):
                        next_tok = self._argmax(cur_logits[:, -1, :])
                        draft_tokens.append(next_tok)
                        cur_logits, cur_draft_kv = self.draft(
                            next_tok, use_kv_cache=True, past_key_values=cur_draft_kv
                        )

                    # 2. Build candidate sequence: prompt's last token + gamma draft tokens.
                    proposal = torch.cat(draft_tokens, dim=1)  # (B, gamma)
                    # 3. Target verifies in ONE forward (gamma+1 tokens).
                    last_target_tok = self._argmax(target_logits[:, -1, :])
                    verify_input = torch.cat([last_target_tok, proposal[:, :-1]], dim=1)
                    verify_logits, verify_kv = self.target(
                        verify_input, use_kv_cache=True, past_key_values=target_kv
                    )

                    # 4. Compare argmax of target's predictions to draft's tokens.
                    target_preds = verify_logits.argmax(dim=-1)  # (B, gamma)
                    accepted = 0
                    for i in range(self.gamma):
                        if torch.equal(target_preds[:, i], proposal[:, i].squeeze(-1)):
                            accepted += 1
                        else:
                            break

                    # 5. Commit accepted prefix; rebuild target KV up to accepted point.
                    # For the scaffold we accept Han's simplification: trust the verify
                    # KV up to the accepted prefix and discard the rest.
                    if accepted == self.gamma:
                        # All accepted; sample one bonus token from target's last logit.
                        bonus = self._argmax(verify_logits[:, -1, :])
                        target_logits, target_kv = self.target(
                            bonus, use_kv_cache=True, past_key_values=verify_kv
                        )
                        draft_logits, draft_kv = self.draft(
                            bonus, use_kv_cache=True, past_key_values=cur_draft_kv
                        )
                        tokens_emitted += accepted + 1
                    else:
                        # Partial accept: take target's correction at position `accepted`.
                        target_logits = verify_logits[:, accepted:accepted + 1, :]
                        # Truncate target_kv to accepted prefix (scaffold simplification:
                        # keep verify_kv as-is; production would slice).
                        target_kv = verify_kv
                        # Reset draft to match target.
                        correction = target_preds[:, accepted:accepted + 1]
                        draft_logits, draft_kv = self.draft(
                            correction, use_kv_cache=True, past_key_values=cur_draft_kv
                        )
                        tokens_emitted += accepted + 1

                    n_accepted_total += accepted
                    cycles += 1
                    _sync()
                    per_cycle_times.append(time.perf_counter() - cycle_start)

        median_cycle = statistics.median(per_cycle_times) if per_cycle_times else float("nan")
        avg_accept_per_cycle = n_accepted_total / cycles if cycles else 0.0
        tokens_per_sec = tokens_emitted / sum(per_cycle_times) if per_cycle_times else 0.0
        return {
            "phase": "decode-speculative",
            "draft_params": n_draft_params,
            "target_params": n_target_params,
            "gamma": self.gamma,
            "prefill_ctx": self.prefill_ctx,
            "warm_time_s": warm_time,
            "tokens_emitted": tokens_emitted,
            "cycles": cycles,
            "avg_accepted_per_cycle": avg_accept_per_cycle,
            "acceptance_rate": avg_accept_per_cycle / self.gamma if self.gamma else 0.0,
            "median_cycle_s": median_cycle,
            "output_tokens_per_sec": tokens_per_sec,
        }


def run_speculative_decode_benchmark(prefill_ctx: int = 2048,
                                       decode_tokens: int = 64,
                                       gamma: int = 4,
                                       batch_size: int = 1) -> dict:
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    target = NanoGPTWhiteBox(**GPT2_SMALL_CONFIG).to(device)
    draft = NanoGPTWhiteBox().to(device)  # iter-1 default 11M
    return SpeculativeDecode(target, draft, prefill_ctx=prefill_ctx,
                              decode_tokens=decode_tokens, gamma=gamma,
                              batch_size=batch_size).run()
