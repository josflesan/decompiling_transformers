import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from typing import Union, Tuple

class ComponentMaskSampler(nn.Module):
    """
    Component Mask Sampler for Stage 1 Pruning
    """
    
    def __init__(self, config):
        super().__init__()
        
        param_idx = 0
        mapping_to_param_idx = {}
        input_vertex = set()
        output_vertex = set()
        
        def walk(node, path):
            nonlocal param_idx
            
            if isinstance(node, dict):
                for k, v in node.items():
                    walk(v, path + (k,))
            elif isinstance(node, list):
                output_vertex.add(path)
                for item in node:
                    mapping_to_param_idx[path + (item,)] = param_idx
                    param_idx += 1
                    input_vertex.add(item)
            else:
                raise RuntimeError(f"Unexpected type encountered when flattening: {type(node)}")
        
        # Flatten the nested configuration of circuit edges paths
        walk(config, ())
        
        self.config = config
        self.mapping_to_param_idx = mapping_to_param_idx
        
        # Initialize masking parameters, source nodes and output nodes
        # input_vertex tells us how many Optimal Ablation vectors to learn
        # output_vertex tells us how to learn global bias term and LayerNorm variance scalar
        self.sample_params = nn.Parameter(torch.ones(param_idx))  # Define masking parameters
        self.input_vertex = list(input_vertex)
        self.output_vertex = list(output_vertex)
        
        assert not any("attention_bias" in item for item in input_vertex)
        
        num_head_per_layer = []
        for i in range(len(config) - 1):
            num_head_per_layer.append(len(config[i]["k"]))
        self.num_head_per_layer = num_head_per_layer
        
        self.window_function = lambda x: x * (1 - x)
    
    def sample_masks(self, bz: int) -> torch.Tensor:
        """
        Creates differentiable masks for training. Continuous relaxation trick to achieve
        differentiability

        Args:
            bz (int): batch size - indicates how many independent random samples of masks to draw

        Returns:
            torch.Tensor: differentiable masks used during training
        """
        
        # 1. Compute base probability for each path to remain by using sigmoid
        prob = F.sigmoid(self.sample_params).unsqueeze(0).expand(bz, -1)
        
        # 2. Generate uniform random noise and define a window where noise is allowed to perturb probability.
        # Stochastic noise makes the masks non-deterministic and encourages exploration
        unif = torch.rand_like(prob)
        window_size = self.window_function(prob).detach()
        prob = window_size * prob + (1 - window_size) * prob.detach()
        masks = ((prob - unif) / window_size + 0.5).clamp(0, 1)
        
        # 3. Force edge masks to 0 if edge destination is severed from rest of network
        masks = self.prune_dangling_edges(masks)
        
        # 4. Computational trick - if mask is pushed to exactly 0/1 (would get grad=0), we reweight the
        # masks to ensure the optimizer receives a useful gradient signal
        n_samples = ((masks < 1 - 1e-3) & (masks > 1e-3)).sum(dim=0).float()
        grad_wts = torch.where(n_samples < 1, 0, bz / n_samples)
        masks = grad_wts * masks + (1 - grad_wts) * masks.detach()
        
        return masks
    
    def prune_dangling_edges(self, masks: torch.Tensor):
        """
        This function removes dangling edges that can result after pruning intermediate edges. As an example, consider
        the general flow of the example path:
        
        Token Embeddings -> L0 Attention -> L1 MLP -> LM Head
        
        If we prune the edge between L0 Attention and L1 MLP, all connections feeding into L0 Attention become useless. They
        need to be pruned not only because otherwise we would waste parameters but we would also miscalculate the L1 sparsity
        penalty. We prune these edges using a two-pass method:
        
        1. Forward Pass (Input Reachability): starting at (WTE, WPE), step forward through the layers. If a layer's Q, K and V
        combine into a positive mask (>0) then the layer's output is marked as reachable. Any edge identified as unreachable is
        forced to 0
        
        2. Backward Pass (Output Reachability): starting from LM Head, work backwards through the layers. If LM Head does not
        receive data from L1 MLP (mask=0), then L1 MLP is marked as useless

        Args:
            masks (torch.Tensor): original masks before being zeroed out if edge is severed
        """
        bz = masks.size(0)
        
        def edge_active(idx):
            return masks[:, idx] > 0
        
        def zero_edge(idx, condition): 
            masks[:, idx].masked_fill_(condition, 0)
        
        with torch.device(masks.device):
            
            with torch.no_grad():
        
                # --------------------------------------------------------------------------------
                # 1) FORWARD PASS - Reachability from inputs
                # --------------------------------------------------------------------------------
                
                reachable_to_input = {
                    "wte": torch.ones(bz, dtype=torch.bool, device=masks.device),
                    "wpe": torch.ones(bz, dtype=torch.bool, device=masks.device),
                }
                
                for layer, num_heads in enumerate(self.num_head_per_layer):
                    # ----- Attention Heads
                    for head in range(num_heads):
                        head_reach = []
                        
                        for act in ["q", "k", "v"]:
                            inputs = self.config[layer][act][head]
                            reach = torch.zeros(bz, dtype=torch.bool, device=masks.device)
                            
                            for inp in inputs:
                                idx = self.mapping_to_param_idx[(layer, act, head, inp)]
                                reach |= edge_active(idx) & reachable_to_input[inp]
                            
                            head_reach.append(reach)
                        
                        reachable_to_input[f"attn_output-{layer}-{head}"] = torch.stack(head_reach).all(dim=0)

                    # ----- MLP
                    mlp_reach = torch.zeros(bz, dtype=torch.bool, device=masks.device)
                    
                    for inp in self.config[layer]['mlp']:
                        idx = self.mapping_to_param_idx[(layer, "mlp", inp)]
                        mlp_reach |= edge_active(idx) & reachable_to_input[inp]

                    reachable_to_input[f"mlp-{layer}"] = mlp_reach
        
            # --------------------------------------------------------------------------------
            # 2) PRUNE EDGES USING INPUT REACHABILITY
            # --------------------------------------------------------------------------------
            
            for layer, num_heads in enumerate(self.num_head_per_layer):
                for head in range(num_heads):
                    out_node = f"attn_output-{layer}-{head}"
                    dangling_out = ~reachable_to_input[out_node]
                    
                    for act in ["q", "k", "v"]:
                        for inp in self.config[layer][act][head]:
                            idx = self.mapping_to_param_idx[(layer, act, head, inp)]
                            dangling = dangling_out | ~reachable_to_input[inp]
                            zero_edge(idx, dangling)
                
                mlp_out = f"mlp-{layer}"
                dangling_out = ~reachable_to_input[mlp_out]
                
                for inp in self.config[layer]["mlp"]:
                    idx = self.mapping_to_param_idx[(layer, "mlp", inp)]
                    dangling = dangling_out | ~reachable_to_input[inp]
                    zero_edge(idx, dangling)
            
            for inp in self.config["lm_head"]:
                idx = self.mapping_to_param_idx[("lm_head", inp)]
                zero_edge(idx, ~reachable_to_input[inp])
            
            with torch.no_grad():
            
                # --------------------------------------------------------------------------------
                # 3) BACKWARD PASS - Reachability to outputs
                # --------------------------------------------------------------------------------
                
                reachable_to_output = defaultdict(
                    lambda: torch.zeros(bz, dtype=torch.bool, device=masks.device)
                )
                
                # Start from LM Head
                for inp in self.config["lm_head"]:
                    idx = self.mapping_to_param_idx[("lm_head", inp)]
                    reachable_to_output[inp] |= edge_active(idx)
                
                for layer in reversed(range(len(self.num_head_per_layer))):
                    
                    # ----- MLP
                    mlp_node = f"mlp-{layer}"
                    
                    for inp in self.config[layer]["mlp"]:
                        idx = self.mapping_to_param_idx[(layer, "mlp", inp)]
                        reachable_to_output[inp] |= (
                            edge_active(idx) & reachable_to_output[mlp_node]
                        )
                    
                    # ----- Attention
                    for head in range(self.num_head_per_layer[layer]):
                        head_node = f"attn_output-{layer}-{head}"
                        reach_head = reachable_to_output[head_node]
                        
                        for act in ["q", "k", "v"]:
                            for inp in self.config[layer][act][head]:
                                idx = self.mapping_to_param_idx[(layer, act, head, inp)]
                                reachable_to_output[inp] |= edge_active(idx) & reach_head
                
            # --------------------------------------------------------------------------------
            # 4) PRUNE EDGES USING OUTPUT REACHABILITY
            # --------------------------------------------------------------------------------
            
            for layer, num_heads in enumerate(self.num_head_per_layer):
                for head in range(num_heads):
                    out_node = f"attn_output-{layer}-{head}"
                    dangling_out = ~reachable_to_output[out_node]
                    
                    for act in ["q", "k", "v"]:
                        for inp in self.config[layer][act][head]:
                            idx = self.mapping_to_param_idx[(layer, act, head, inp)]
                            dangling = dangling_out | ~reachable_to_output[inp]
                            zero_edge(idx, dangling)
                
                mlp_out = f"mlp-{layer}"
                dangling_out = ~reachable_to_output[mlp_out]
                
                for inp in self.config[layer]["mlp"]:
                    idx = self.mapping_to_param_idx[(layer, "mlp", inp)]
                    dangling = dangling_out | ~reachable_to_output[inp]
                    zero_edge(idx, dangling)
            
            for inp in self.config["lm_head"]:
                idx = self.mapping_to_param_idx[("lm_head", inp)]
                zero_edge(idx, ~reachable_to_output[inp])
            
        return masks
    
    def get_penalty(self, node_reg_coef: float) -> Union[Tuple[float, Tuple[float, int]], None]:
        """
        Function which computes the sparsity penalty term used in the loss function. Sigmoid computes
        the expected active probability of each edge and summing them up gives us our sparsity.

        Args:
            node_reg_coef (float): used to support node-level regularization

        Returns:
            Tuple[torch.Tensor, Tuple[float]]: the first value is the L1 penalty tensor, second value
            are the actual penalty floats (detached) meant for logging
        """
        
        edge_reg = F.sigmoid(self.sample_params).sum()
        
        if node_reg_coef == 0:
            return edge_reg, (edge_reg.item(), 0)
        else:
            raise NotImplementedError()
    
    def sample_binary_masks(self, bz: int, threshold: float=0) -> torch.Tensor:
        """
        Method used for testing/evaluation when we don't want stochastic masks but a hard architectural
        commitment. We use threshold to determine when to zero out an edge. We prune dangling edges
        once more before passing final graph

        Args:
            bz (int): batch size, indicates how to restructure the flat nn.Parameter
            threshold (float, optional): _description_. Defaults to 0.

        Returns:
            torch.Tensor: the final set of binary masks to be used for pruning during evaluation
        """
        
        masks = (self.sample_params > threshold).float().unsqueeze(0).expand(bz, -1)
        masks = self.prune_dangling_edges(masks)
        return masks

class FullPathsMaskSampler(ComponentMaskSampler):
    """Path Mask Sampler for Pruning Stage 2"""
    
    def __init__(self, config, split_mlp):
        super().__init__(config)
        
        output_vertex = set(self.output_vertex)
        v_output_vertex = set()
        mlp_output_vertex = set()
        
        # Config transformation stage 2
        for k1 in config:
            if type(config[k1]) == dict:
                for k2 in config[k1]:
                    
                    # If this is an attention value node, convert component
                    # into path from sending vertices to receiving vertices of
                    # this node
                    if k2 == "v":
                        for k3 in config[k1][k2]:
                            output_vertex.remove((k1, k2, k3))
                            for item in config[k1][k2][k3]:
                                v_output_vertex.add((k1, k2, k3, item))
                    
                    # If this is an MLP node and we are splitting, remove original
                    # MLP node from the config and convert to series of paths
                    # to each receiving vertex
                    elif split_mlp and k2 == "mlp":
                        output_vertex.remover((k1, k2))
                        for item in config[k1][k2]:
                            mlp_output_vertex.add((k1, k2, item))
        
        
        self.output_vertex = list(output_vertex)
        self.all_output_vertex = list(output_vertex.union(v_output_vertex).union(mlp_output_vertex))
        self.mlp_output_vertex = list(mlp_output_vertex)
        self.split_mlp = split_mlp
        
        self.sample_params.data = torch.ones_like(self.sample_params.data)
    
    def prune_dangling_edges(self, masks):
        with torch.device(masks.device):
            
            bz = masks.size(0)
            
            def edge_active(idx):
                return masks[:, idx] > 0
        
            def zero_edge(idx, dangling):
                masks[:, idx].masked_fill_(dangling, 0)
            
            with torch.no_grad():
                
                # --------------------------------------------------------------------------------
                # 1) FORWARD PASS - Reachability from inputs
                # --------------------------------------------------------------------------------
                
                reachable_to_input = {
                    "wte": torch.ones(bz, dtype=torch.bool, device=masks.device),
                    "wpe": torch.ones(bz, dtype=torch.bool, device=masks.device)
                }
                
                for layer, num_heads in enumerate(self.num_head_per_layer):
                    
                    # ----- Attention Heads
                    for head in range(num_heads):
                        reach_qk = []
                        
                        for act in ["q", "k"]:
                            reach = [torch.zeros(bz, dtype=torch.bool, device=masks.device)]
                            for input_v in self.config[layer][act][head]:
                                reach.append(edge_active(self.mapping_to_param_idx[(layer, act, head, input_v)]) & reachable_to_input[input_v])
                            reach = torch.stack(reach).any(dim=0)
                            reach_qk.append(reach)
                        
                        for input_v in self.config[layer]["v"][head]:
                            reach = edge_active(self.mapping_to_param_idx[(layer, "v", head, input_v)] & reachable_to_input[input_v])
                            attn_reach = reach_qk[0] & reach_qk[1] & reach
                            reachable_to_input[f"attn_output-{layer}-{head}-{input_v}"] = attn_reach

                    
                    # ----- MLP
                    if not self.split_mlp:
                        
                        # If any of the individual split MLPs are reachable from input, 
                        mlp_reach = [torch.zeros(bz, dtype=torch.bool, device=masks.device)]
                        for input_v in self.config[layer]["mlp"]:
                            mlp_reach.append(edge_active(self.mapping_to_param_idx[(layer, "mlp", input_v)]) & reachable_to_input[input_v])
                        mlp_reach = torch.stack(mlp_reach).any(dim=0)
                        
                        reachable_to_input[f"mlp-{layer}"] = mlp_reach
                    else:
                        for input_v in self.config[layer]["mlp"]:
                            mlp_reach = edge_active(self.mapping_to_param_idx[(layer, "mlp", input_v)]) & reachable_to_input[input_v]
                            reachable_to_input[f"mlp-{layer}-{input_v}"] = mlp_reach
            
            # --------------------------------------------------------------------------------
            # 2) PRUNE EDGES USING INPUT REACHABILITY
            # --------------------------------------------------------------------------------
            for layer, num_heads in enumerate(self.num_head_per_layer):
                for head in range(num_heads):
                    
                    # Collect reachability to input for each head's sending vertex
                    dangling_head = [torch.ones(bz, dtype=torch.bool(), device=masks.device)]
                    
                    # Consider value inputs and their output nodes first
                    for input_v in self.config[layer]["v"][head]:
                        out_node = f"attn_output-{layer}-{head}-{input_v}"
                        dangling_out = ~reachable_to_input[out_node]
                        dangling_head.append(dangling_out)
                        
                        # If either full path node is not reachable or the head output not reachable,
                        # the path is dangling
                        dangling = dangling_out | ~reachable_to_input[input_v]
                        zero_edge(self.mapping_to_param_idx[(layer, "v", head, input_v)], dangling)
                    
                    # If the head output is fully "dead" we also zero out query and key paths (zero out the whole head)
                    dangling_out = torch.stack(dangling_head).all(dim=0)
                    for activation in ["q", "k"]:
                        for input_v in self.config[layer][activation][head]:
                            dangling = dangling_out | ~reachable_to_input[input_v]
                            zero_edge(self.mapping_to_param_idx[(layer, activation, head, input_v)], dangling)
                
                # For each MLP output, prune if either full path node or output is not reachable
                for input_v in self.config[layer]["mlp"]:
                    out_node = f"mlp-{layer}" if not self.split_mlp else f"mlp-{layer}-{input_v}"
                    dangling_out = ~reachable_to_input[out_node]
                    dangling = dangling_out | ~reachable_to_input[input_v]
                    zero_edge(self.mapping_to_param_idx[(layer, "mlp", input_v)], dangling)
            
            # For each output of the LM Head, zero out if not reachable
            for input_v in self.config["lm_head"]:
                dangling = ~reachable_to_input[input_v]
                zero_edge(self.mapping_to_param_idx[("lm_head", input_v)], dangling)
            
            with torch.no_grad():
                # --------------------------------------------------------------------------------
                # 3) BACKWARD PASS - Reachability to outputs
                # --------------------------------------------------------------------------------
                
                reachable_to_output = defaultdict(
                    lambda: torch.zeros(bz, dtype=torch.bool, device=masks.device)
                )
                
                # Start from LM Head
                for input_v in self.config["lm_head"]:
                    reachable_to_output[input_v] = edge_active(self.mapping_to_param_idx[("lm_head", input_v)])
                
                for layer in reversed(range(len(self.num_head_per_layer))):
                    
                    # ------- MLP nodes
                    for input_v in self.config[layer]["mlp"]:
                        # If the MLP node can reach the output and the input edge is active, path exists
                        mlp_node = f"mlp-{layer}" if not self.split_mlp else f"mlp-{layer}-{input_v}"
                        reach = edge_active(self.mapping_to_param_idx[(layer, "mlp", input_v)]) & reachable_to_output[mlp_node]
                        reachable_to_output[input_v] = reachable_to_output[input_v] | reach
                    
                    # ------- Attention Heads
                    for head in range(self.num_head_per_layer[layer]):
                        
                        # Determine whether each head can reach the output
                        head_reach = [torch.zeros(bz, dtype=torch.bool, device=masks.device)]
                        for input_v in self.config[layer]["v"][head]:
                            # If the head can reach the output and input edge exists, path exists
                            head_reach.append(reachable_to_output[f"attn_output-{layer}-{head}-{input_v}"])
                            reach = edge_active(self.mapping_to_param_idx[(layer, "v", head, input_v)]) & reachable_to_output[f"attn_output-{layer}-{head}-{input_v}"]
                            reachable_to_output[input_v] = reachable_to_output[input_v] | reach
                        
                        # If any head is completely dead, mark query and key paths as dead too
                        head_reach = torch.stack(head_reach).any(dim=0)
                        for act in ["q", "k"]:
                            
                            for input_v in self.config[layer][act][head]:
                                reach = edge_active(self.mapping_to_param_idx[(layer, act, head, input_v)]) & head_reach
                                reachable_to_output[input_v] = reachable_to_output[input_v] | reach
            
            # --------------------------------------------------------------------------------
            # 4) PRUNE EDGES USING OUTPUT REACHABILITY
            # --------------------------------------------------------------------------------
            for layer, num_head in enumerate(self.num_head_per_layer):
                for head in range(num_head):
                    # Handle value outputs first
                    dangling_head = [torch.ones(bz, dtype=torch.bool, device=masks.device)]
                    
                    for input_v in self.config[layer]['v'][head]:
                        v_name = f"attn_output-{layer}-{head}-{input_v}"
                        dangling_out = ~reachable_to_output[v_name]
                        dangling_head.append(dangling_out)
                        
                        dangling = dangling_out | ~reachable_to_output[input_v]
                        zero_edge(self.mapping_to_param_idx[(layer, "v", head, input_v)], dangling)
                    
                    # If head is completely dead (does not output anything)
                    # Remove all query and key nodes pertaining to it
                    dangling_out = torch.stack(dangling_head).all(dim=0)
                    for activation in ["q", "k"]:
                        for input_v in self.config[layer][activation][head]:
                            dangling = dangling_out | ~reachable_to_output[input_v]
                            zero_edge(self.mapping_to_param_idx[(layer, activation, head, input_v)], dangling)
                
                # For each MLP, if it cannot reach output, remove
                for input_v in self.config[layer]["mlp"]:
                    mlp_name = f"mlp-{layer}" if not self.split_mlp else f"mlp-{layer}-{input_v}"
                    dangling_out = ~reachable_to_output[mlp_name]
                    dangling = dangling_out | ~reachable_to_output[input_v]
                    zero_edge(self.mapping_to_param_idx[(layer, "mlp", input_v)], dangling)
            
            # If the LM head cannot reach output, remove
            for input_v in self.config["lm_head"]:
                dangling = ~reachable_to_output[input_v]
                zero_edge(self.mapping_to_param_idx[("lm_head", input_v)], dangling)
            
        return masks