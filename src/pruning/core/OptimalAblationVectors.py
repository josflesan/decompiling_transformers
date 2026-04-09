import torch
import torch.nn as nn

from transformers.models.gpt2.modeling_gpt2 import GPT2MLP

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
        
        # Input vertex OA corresponds to the optimal ablation vector used if input to a vertex is pruned
        self.input_vertex = input_vertex
        self.input_vertex_oa = nn.Parameter(torch.zeros(len(input_vertex), d_model))
        self.to_in_oa_idx = {item: i for i, item in enumerate(input_vertex)}
        
        # Output vertex OA corresponds to the optimal ablation vector used as output when vertex is pruned
        self.output_vertex = output_vertex
        self.output_vertex_oa = nn.Parameter(torch.zeros(len(output_vertex), d_model))
        self.to_out_oa_idx = {item: i for i, item in enumerate(output_vertex)}
        
        # LN Vertex corresponds to the layer normalization gamma being learned
        assert init_var.size(0) == len(ln_vertex)
        self.ln_vertex = ln_vertex
        self.ln_var = nn.Parameter(init_var)
        self.to_ln_idx = {item: i for i, item in enumerate(ln_vertex)}
        
        # MLP Vertex corresponds to the single-dependency MLPs enabled by the split_mlp flag in stage 2 pruning
        self.mlp_vertex = mlp_vertex
        if mlp_vertex is not None and len(mlp_vertex) > 0:
            inner_dim = model_config.n_inner if model_config.n_inner is not None else 4 * model_config.hidden_size
            self.mlps = nn.ModuleDict({f"{item[0]} {item[2]}": GPT2MLP(inner_dim, model_config) for item in mlp_vertex})  # item: (0, 'mlp', 'attn_output-1-0-attn_output-0-0-wte')

class OptimalQueryBiasVectors(nn.Module):
    """
    Trainable optimal ablation vectors for each of the prunable elements of stage 3. In particular:

    1. q_bias_term - when we want to prune a query from our resulting sum of products, that query needs
                    to be replaced by a learned vector which can interact with each key
    2. output_vertex_oa - optimal ablation vectors used when a component is pruned. Needed because if we do
                    further pruning, the previous optimal vectors will no longer be optimal
    3. ln_vertex - learned linearized layer norm from previous stages
    4. mlp_vertex - learned split MLPs from previous stages
    """
    
    def __init__(self, key_names, d_head, oa_vecs: OptimalAblationVectors):
        super().__init__()
        self.key_names = key_names
        self.q_bias_term = nn.Parameter(torch.zeros(len(key_names), d_head))
        self.to_q_bias = {item: i for i, item in enumerate(key_names)}
        
        # Recover optimal ablation vectors from previous stages
        # Note we do not need optimal ablation vertices for deep paths, only for direct inputs to prunable QK vertices
        self.output_vertex = [item for item in oa_vecs.output_vertex if len(item) <= 2] # Only needed for MLP if not split
        self.to_out_oa_idx = {item: i for i, item in enumerate(self.output_vertex)}
        output_vertex_oa = torch.zeros(len(self.output_vertex), oa_vecs.output_vertex_oa.size(1))
        for item in self.output_vertex:
            output_vertex_oa[self.to_out_oa_idx[item]] = oa_vecs.output_vertex_oa[oa_vecs.to_out_oa_idx[item]].data
        self.output_vertex_oa = nn.Parameter(output_vertex_oa)
        
        self.ln_vertex = oa_vecs.ln_vertex
        self.ln_var = oa_vecs.ln_var
        self.to_ln_idx = oa_vecs.to_ln_idx
        
        self.mlp_vertex = oa_vecs.mlp_vertex
        if hasattr(oa_vecs, "mlps"):
            self.mlps = oa_vecs.mlps