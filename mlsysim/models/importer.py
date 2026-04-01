import json
import urllib.request
import urllib.error
import time
import warnings
from typing import Optional
import logging
from .types import TransformerWorkload
from ..core.constants import ureg

logger = logging.getLogger(__name__)

def fetch_hf_config(model_id: str, max_retries: int = 3, timeout: int = 10) -> dict:
    """Fetches the config.json from Hugging Face with retries and timeout."""
    config_url = f"https://huggingface.co/{model_id}/resolve/main/config.json"
    req = urllib.request.Request(config_url, headers={'User-Agent': 'mlsysim/1.0'})
    
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                data = response.read().decode('utf-8')
                try:
                    return json.loads(data)
                except json.JSONDecodeError as je:
                    raise ValueError(f"Invalid JSON in config for '{model_id}': {je}")
        except urllib.error.HTTPError as e:
            if e.code == 429: # Rate limit
                wait_time = (2 ** attempt) + 1
                logger.warning(f"Rate limited by Hugging Face (429). Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            elif e.code == 404:
                raise ValueError(f"Model '{model_id}' not found on Hugging Face or is private. (404 Not Found)")
            else:
                raise ValueError(f"HTTP Error {e.code} while fetching config for '{model_id}': {e.reason}")
        except urllib.error.URLError as e:
            if 'CERTIFICATE_VERIFY_FAILED' in str(e.reason):
                import ssl
                warnings.warn(
                    "SSL verification disabled for HuggingFace API. This is insecure.",
                    stacklevel=2,
                )
                context = ssl._create_unverified_context()
                try:
                    with urllib.request.urlopen(req, context=context, timeout=timeout) as response:
                        data = response.read().decode('utf-8')
                        try:
                            return json.loads(data)
                        except json.JSONDecodeError as je:
                            raise ValueError(f"Invalid JSON in config for '{model_id}': {je}")
                except ValueError:
                    raise
                except Exception as inner_e:
                    raise ValueError(f"SSL fallback failed for '{model_id}': {inner_e}")
            
            logger.warning(f"Network error (attempt {attempt + 1}/{max_retries}): {e.reason}")
            time.sleep((2 ** attempt))
        except Exception as e:
            raise ValueError(f"Unexpected error fetching config for '{model_id}': {e}")
            
    raise TimeoutError(f"Failed to fetch config for '{model_id}' after {max_retries} attempts.")


def import_hf_model(model_id: str, name: Optional[str] = None) -> TransformerWorkload:
    """
    Imports a model configuration directly from Hugging Face Hub 
    and converts it into an mlsysim TransformerWorkload.
    
    This avoids heavy dependencies like `transformers` or `torch` by directly 
    fetching and parsing the `config.json` file.
    
    Args:
        model_id: The Hugging Face model ID (e.g., 'meta-llama/Meta-Llama-3-8B')
        name: Optional custom name for the workload. Defaults to the model ID repo name.
        
    Returns:
        TransformerWorkload: An mlsysim-compatible workload object.
    """
    config = fetch_hf_config(model_id)

    # Standardize architecture fields across different HF model types (Llama, GPT-NeoX, Bert, etc.)
    layers = config.get("num_hidden_layers") or config.get("n_layer") or config.get("num_layers")
    hidden_dim = config.get("hidden_size") or config.get("n_embd") or config.get("d_model")
    heads = config.get("num_attention_heads") or config.get("n_head") or config.get("num_heads")
    kv_heads = config.get("num_key_value_heads") or config.get("multi_query_group_num") or heads
    
    if not all([layers, hidden_dim, heads]):
        missing = [k for k, v in zip(["layers", "hidden_dim", "heads"], [layers, hidden_dim, heads]) if not v]
        raise ValueError(f"Could not parse required architecture fields from {model_id} config.json. Missing: {missing}")

    vocab_size = config.get("vocab_size", 32000)
    intermediate_size = config.get("intermediate_size") 
    
    # Analytical Parameter Estimation
    # This is a first-principles estimation of parameter count.
    head_dim = hidden_dim // heads
    attn_params = hidden_dim * hidden_dim + 2 * (hidden_dim * head_dim * kv_heads) + hidden_dim * hidden_dim
    
    if intermediate_size:
        # Many modern LLMs (Llama, Mistral) use SwiGLU: 3 matrices (gate, up, down)
        ffn_params = 3 * hidden_dim * intermediate_size
    else:
        # Standard GPT/BERT use 2 matrices, typically with 4x expansion
        ffn_params = 2 * hidden_dim * (4 * hidden_dim)

    layer_params = attn_params + ffn_params
    
    tied_embeddings = config.get("tie_word_embeddings", False)
    if tied_embeddings:
        total_params = (vocab_size * hidden_dim) + (layers * layer_params)
    else:
        total_params = 2 * (vocab_size * hidden_dim) + (layers * layer_params)

    resolved_name = name or model_id.split("/")[-1]

    # Calculate theoretical max FLOPs per token during forward pass (inference)
    # Roughly 2 FLOPs per parameter
    inference_flops = 2 * total_params * ureg.flop

    return TransformerWorkload(
        name=resolved_name,
        architecture="Transformer",
        parameters=total_params * ureg.param,
        layers=layers,
        hidden_dim=hidden_dim,
        heads=heads,
        kv_heads=kv_heads,
        inference_flops=inference_flops
    )
