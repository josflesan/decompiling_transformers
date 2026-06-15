from transformer_lens.model_bridge import TransformerBridge
from typing import Dict, List

def get_all_hook_sites(model: TransformerBridge) -> Dict[str, List[str]]:
    """Enumerate all patchable hook sites by category"""
    
    num_layers = model.cfg.n_layers
    num_heads = model.cfg.n_heads
    
    return {
        "residual_pre": [f"blocks.{l}.hook_resid_pre" for l in range(num_layers)],
        "residual_post": [f"blocks.{l}.hook_resid_post" for l in range(num_layers)],
        "attention_output": [f"attn_output-{l}-{h}" for l in range(num_layers) for h in range(num_heads)],
        "mlp": [f"blocks.{l}.hook_mlp_out" for l in range(num_layers)],
        "qkv": [(f"blocks.{l}.attn.q", f"blocks.{l}.attn.k", f"blocks.{l}.attn.v") for l in range(num_layers)],
    }