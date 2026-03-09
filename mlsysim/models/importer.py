import json
import urllib.request
from typing import Optional
from .types import TransformerWorkload
from ..core.constants import ureg

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
    config_url = f"https://huggingface.co/{model_id}/resolve/main/config.json"
    
    try:
        req = urllib.request.Request(config_url, headers={'User-Agent': 'mlsysim/1.0'})
        with urllib.request.urlopen(req) as response:
            config = json.loads(response.read().decode())
    except Exception as e:
        if 'CERTIFICATE_VERIFY_FAILED' in str(e):
            import ssl
            context = ssl._create_unverified_context()
            with urllib.request.urlopen(req, context=context) as response:
                config = json.loads(response.read().decode())
        else:
            raise ValueError(f"Failed to fetch config for {model_id}. Ensure the model is public and the ID is correct. Error: {e}")

    # Standardize architecture fields across different HF model types (Llama, GPT-NeoX, Bert, etc.)
    layers = config.get("num_hidden_layers") or config.get("n_layer") or config.get("num_layers")
    hidden_dim = config.get("hidden_size") or config.get("n_embd") or config.get("d_model")
    heads = config.get("num_attention_heads") or config.get("n_head") or config.get("num_heads")
    kv_heads = config.get("num_key_value_heads") or config.get("multi_query_group_num") or heads
    
    if not all([layers, hidden_dim, heads]):
        raise ValueError(f"Could not parse required architecture fields from {model_id} config.json: layers={layers}, hidden={hidden_dim}, heads={heads}")

    vocab_size = config.get("vocab_size", 32000)
    intermediate_size = config.get("intermediate_size") 
    
    # Analytical Parameter Estimation
    # This is a first-principles estimation of parameter count.
    # We estimate:
    # 1. Embedding Layer: V * H
    # 2. Attention (Q, K, V, O matrices):
    #    Q = H * (H/heads * heads) = H^2
    #    K, V = H * (H/heads * kv_heads)
    #    O = H^2
    head_dim = hidden_dim // heads
    attn_params = hidden_dim * hidden_dim + 2 * (hidden_dim * head_dim * kv_heads) + hidden_dim * hidden_dim
    
    # 3. FFN (Feed-Forward Network):
    if intermediate_size:
        # Many modern LLMs (Llama, Mistral) use SwiGLU: 3 matrices (gate, up, down)
        ffn_params = 3 * hidden_dim * intermediate_size
    else:
        # Standard GPT/BERT use 2 matrices, typically with 4x expansion
        ffn_params = 2 * hidden_dim * (4 * hidden_dim)

    layer_params = attn_params + ffn_params
    
    # 4. Total params: Embeddings + Layers + Output Head (often tied)
    tied_embeddings = config.get("tie_word_embeddings", False)
    if tied_embeddings:
        total_params = (vocab_size * hidden_dim) + (layers * layer_params)
    else:
        total_params = 2 * (vocab_size * hidden_dim) + (layers * layer_params)

    resolved_name = name or model_id.split("/")[-1]

    return TransformerWorkload(
        name=resolved_name,
        architecture="Transformer",
        parameters=total_params * ureg.param,
        layers=layers,
        hidden_dim=hidden_dim,
        heads=heads,
        kv_heads=kv_heads,
        inference_flops=2 * total_params * ureg.flop
    )
