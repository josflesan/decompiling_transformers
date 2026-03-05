import torch
import torch.nn as nn

from transformers.model.gpt2.modeling_gpt2 import GPT2MLP

class OptimalAblationVectors(nn.Module):
    """
    Trainable optimal ablation vectors for each of the types of nodes/variables we aim to learn. In particular:
    
    1. input_vertex - source node, optimal vectors to be replaced as the generic signal when a path/component
                    is pruned from the computation graph
    2. output_vertex - destination nodes, optimal vectors that act as global biases for the destination nodes. Used
                    to stabilize the destination node
    3. ln_vertex - output vertex locations prior to a LayerNorm (gammas)
    4. mlp_vertex - used when splitting MLPs. This holds a set of parallel single-input MLPs which when composed approximate
                    the original MLP network
    """
    
    def __init__(self, input_vertex, output_vertex, ln_vertex, mlp_vertex, model_config, init_var):
        super().__init__()
        d_model = model_config.hidden_size
        
        # Input vertex OA corresponds to the optimal ablation vertex used if pruned
        self.input_vertex = input_vertex
        self.input_vertex_oa = nn.Parameter(torch.zeros(len(input_vertex), d_model))
        self.to_in_oa_idx = {item: i for i, item in enumerate(input_vertex)}
        
        # Output vertex OA corresponds to the optimal bias
        self.output_vertex = output_vertex
        self.output_vertex_oa = nn.Parameter(torch.zeros(len(output_vertex), d_model))
        self.to_out_oa_idx = {item: i for i, item in enumerate(output_vertex)}
        
        # LN Vertex corresponds to the layer normalization gamma being learned
        assert init_var.size(0) == len(ln_vertex)
        self.ln_vertex = ln_vertex
        self.ln_var = nn.Parameter(init_var)
        self.to_ln_idx = {item: i for i, item in enumerate(ln_vertex)}
        
        # MLP Vertex corresponds to... TODO
        self.mlp_vertex = mlp_vertex
        if mlp_vertex is not None and len(mlp_vertex) > 0:
            inner_dim = model_config.n_inner if model_config.n_inner is not None else 4 * model_config.hidden_size
            self.mlps = nn.ModuleDict({f"{item[0]} {item[2]}": GPT2MLP(inner_dim, model_config) for item in mlp_vertex})  # item: (0, 'mlp', 'attn_output-1-0-attn_output-0-0-wte')