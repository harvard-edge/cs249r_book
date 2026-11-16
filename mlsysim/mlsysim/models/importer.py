import json
import logging
import time
import urllib.error
import urllib.request
from typing import Optional

from ..core.units import ureg
from .types import SparseTransformerWorkload, TransformerWorkload

logger = logging.getLogger(__name__)

_NON_GATED_ACTS = frozenset({
    "gelu",
    "gelu_new",
    "gelu_fast",
    "gelu_approx",
    "relu",
    "relu2",
    "tanh",
    "quick_gelu",
})

_NON_GATED_MODEL_TYPES = frozenset({
    "bert",
    "bloom",
    "falcon",
    "gpt2",
    "gpt_bigcode",
    "gpt_neox",
    "gptj",
    "opt",
    "roberta",
})


def fetch_hf_config(model_id: str, max_retries: int = 3, timeout: int = 10) -> dict:
    """Fetch ``config.json`` from Hugging Face Hub with verified TLS, retries, and timeout."""
    config_url = f"https://huggingface.co/{model_id}/resolve/main/config.json"
    req = urllib.request.Request(config_url, headers={"User-Agent": "mlsysim/1.0"})

    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                data = response.read().decode("utf-8")
                try:
                    return json.loads(data)
                except json.JSONDecodeError as je:
                    raise ValueError(f"Invalid JSON in config for '{model_id}': {je}") from je
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait_time = (2 ** attempt) + 1
                logger.warning("Rate limited by Hugging Face (429). Retrying in %ss...", wait_time)
                time.sleep(wait_time)
                continue
            if e.code == 404:
                raise ValueError(
                    f"Model '{model_id}' not found on Hugging Face or is private. (404 Not Found)"
                ) from e
            raise ValueError(
                f"HTTP Error {e.code} while fetching config for '{model_id}': {e.reason}"
            ) from e
        except urllib.error.URLError as e:
            if "CERTIFICATE_VERIFY_FAILED" in str(e.reason):
                raise ValueError(
                    f"TLS certificate verification failed while fetching config for '{model_id}': "
                    f"{e.reason}. Configure SSL_CERT_FILE or install system/certifi CA certificates."
                ) from e
            logger.warning("Network error (attempt %s/%s): %s", attempt + 1, max_retries, e.reason)
            time.sleep(2 ** attempt)
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Unexpected error fetching config for '{model_id}': {e}") from e

    raise TimeoutError(f"Failed to fetch config for '{model_id}' after {max_retries} attempts.")


def _uses_gated_mlp(config: dict, intermediate_size: Optional[int]) -> bool:
    """Determine whether the Transformer FFN uses a 3-matrix gated projection (e.g., SwiGLU)."""
    if config.get("is_gated_act") is True:
        return True
    if config.get("is_gated_act") is False:
        return False
    act = str(config.get("hidden_act") or config.get("activation_function") or "").lower()
    model_type = str(config.get("model_type") or "").lower()
    if act in _NON_GATED_ACTS or model_type in _NON_GATED_MODEL_TYPES:
        return False
    return intermediate_size is not None


def import_hf_model(model_id: str, name: Optional[str] = None) -> TransformerWorkload:
    """Import a Hugging Face Hub ``config.json`` into a ``TransformerWorkload``.

    Parameters
    ----------
    model_id : str
        Hugging Face model identifier (e.g., ``'meta-llama/Meta-Llama-3-8B'``).
    name : str, optional
        Custom workload display name; defaults to the repository basename.

    Returns
    -------
    TransformerWorkload
        ``SparseTransformerWorkload`` for Mixture-of-Experts configurations, or
        dense ``TransformerWorkload`` otherwise.
    """
    config = fetch_hf_config(model_id)

    layers = config.get("num_hidden_layers") or config.get("n_layer") or config.get("num_layers")
    hidden_dim = config.get("hidden_size") or config.get("n_embd") or config.get("d_model")
    heads = config.get("num_attention_heads") or config.get("n_head") or config.get("num_heads")
    kv_heads = config.get("num_key_value_heads") or config.get("multi_query_group_num") or heads

    if not all([layers, hidden_dim, heads]):
        missing = [
            k for k, v in zip(["layers", "hidden_dim", "heads"], [layers, hidden_dim, heads]) if not v
        ]
        raise ValueError(
            f"Could not parse required architecture fields from {model_id} config.json. Missing: {missing}"
        )

    vocab_size = int(config.get("vocab_size", 32000))
    context_length = (
        config.get("max_position_embeddings")
        or config.get("n_positions")
        or config.get("seq_length")
    )
    explicit_head_dim = config.get("head_dim")
    head_dim = int(explicit_head_dim) if explicit_head_dim is not None else (hidden_dim // heads)

    q_width = heads * head_dim
    kv_width = kv_heads * head_dim
    attn_params = (hidden_dim * q_width) + 2 * (hidden_dim * kv_width) + (q_width * hidden_dim)

    intermediate_size = config.get("intermediate_size") or config.get("ffn_dim") or config.get("n_inner")
    ffn_matrices = 3 if _uses_gated_mlp(config, intermediate_size) else 2
    ffn_width = int(intermediate_size) if intermediate_size is not None else (4 * hidden_dim)

    # Hugging Face PretrainedConfig defaults tie_word_embeddings to True when omitted.
    tied_embeddings = bool(config.get("tie_word_embeddings", True))
    embed_params = (vocab_size * hidden_dim) if tied_embeddings else (2 * vocab_size * hidden_dim)

    resolved_name = name or model_id.split("/")[-1]

    num_experts = (
        config.get("num_local_experts")
        or config.get("n_routed_experts")
        or config.get("num_experts")
    )
    if num_experts is not None and int(num_experts) > 1:
        experts = int(num_experts)
        active_experts = int(
            config.get("num_experts_per_tok")
            or config.get("num_selected_experts")
            or 1
        )
        shared_experts = int(config.get("n_shared_experts") or 0)
        moe_width = int(config.get("moe_intermediate_size") or ffn_width)
        expert_ffn = ffn_matrices * hidden_dim * moe_width
        total_layer_params = attn_params + (experts + shared_experts) * expert_ffn
        active_layer_params = attn_params + (active_experts + shared_experts) * expert_ffn
        total_params = embed_params + layers * total_layer_params
        active_params = embed_params + layers * active_layer_params
        return SparseTransformerWorkload(
            name=resolved_name,
            architecture="MoE",
            parameters=total_params * ureg.param,
            active_parameters=active_params * ureg.param,
            experts=experts,
            active_experts_per_token=active_experts,
            layers=layers,
            hidden_dim=hidden_dim,
            heads=heads,
            kv_heads=kv_heads,
            head_dim=int(explicit_head_dim) if explicit_head_dim is not None else None,
            vocab_size=vocab_size,
            context_length=int(context_length) if context_length is not None else None,
            inference_flops=2 * active_params * ureg.flop,
        )

    ffn_params = ffn_matrices * hidden_dim * ffn_width
    total_params = embed_params + layers * (attn_params + ffn_params)

    return TransformerWorkload(
        name=resolved_name,
        architecture="Transformer",
        parameters=total_params * ureg.param,
        layers=layers,
        hidden_dim=hidden_dim,
        heads=heads,
        kv_heads=kv_heads,
        head_dim=int(explicit_head_dim) if explicit_head_dim is not None else None,
        vocab_size=vocab_size,
        context_length=int(context_length) if context_length is not None else None,
        inference_flops=2 * total_params * ureg.flop,
    )
