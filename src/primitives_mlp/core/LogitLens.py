import logging
import re
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Tuple

from data.CustomTokenizer import CustomTokenizer
from primitives_mlp.utilities.activation_tracing import trace_mlp
from primitives_mlp.utilities.mlp_primitive_dataclasses import PrimitiveSearchOutput
from primitives_mlp.utilities.parameter_getters import (
    get_attn_weights_for_head,
    get_ln_matrix_for_node,
    get_input_weights_symbolic,
    get_qk_for_head,
    get_ov_for_head
)
from pruning.core.OptimalAblationVectors import OptimalQueryBiasVectors
from pruning.core.hooks import GPT2QKHooks
from utilities.metrics_logger import MetricsLogger

class LogitLens:
    """
    This class implements the 'correct' LogitLens-like procedure described by the authors
    in their paper. In particular, it applies ('absorbs') all of the transformations appearing
    after the component we want to analyse **before** applying the unembedding layer. This is
    in contrast to regular LogitLens which simply applies the unembedding layer to the residual
    stream as soon as it accumulates the component's contribution.
    
    In the case of QK inspection, ... TODO (I suspect this does not involve the unembedding layer)
    """
    
    def __init__(
        self,
        hooked_model: GPT2QKHooks,
        oa_vecs: OptimalQueryBiasVectors,
        tokenizer: CustomTokenizer,
        dataloader: DataLoader,
        converted_mlp: Dict[str, PrimitiveSearchOutput],
        metrics_logger: MetricsLogger,
        logger: logging.Logger,
        cache_num: int=1000
    ):
        self.logger = logger
        self.metrics_logger = metrics_logger
        
        self.hooked_model = hooked_model
        self.device = self.hooked_model.device
        self.oa_vecs = oa_vecs
        self.tokenizer = tokenizer
        self.dataloader = dataloader
        self.converted_mlp = converted_mlp
        self.cache_num = cache_num
    
    @torch.no_grad()
    def _inspect_mlp(
        self,
        complete_path: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This method returns pairs of MLP inputs and traced outputs for a target (unreplaced) MLP.
        The inputs to the MLP are determined by tracing the batch inputs upstream until the MLP target.
        The outputs are determined by tracing the MLP inputs downstream until the unembedding/attention layer.
        
        This method contains the generic logic for the (input, output) pair composition, and is used by
        other methods in the class in order to obtain pairs for a specific downstream output (either
        unembedding or attention). The effect of the MLP can be investigated by plotting a heatmap of the
        inputs and outputs in the relevant space. For details on this, refer to utilities/heatmap_plotting.py

        Args:
            complete_path (str): the path containing an MLP that failed replacement

        Raises:
            RuntimeError: if there is an unrecognized component in the path downstream from the target MLP

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: inputs, outputs of the target MLP for interpretation
        """
        
        # Find relevant inputs to the MLP based on path
        pattern = r'attn_output-\d+-\d+|mlp-\d+|lm_head|wte|wpe'
        split_nodes = re.findall(pattern, complete_path)
        mlp_path_idx = None
        
        for idx, node in enumerate(split_nodes):
            if node.startswith("mlp") and ("-".join(split_nodes[idx:]) not in self.converted_mlp):
                mlp_path_idx = idx
        
        if mlp_path_idx is None:
            # If there is no unexplained MLP, return
            return None, None

        # Find the path from MLP to the output
        mlp_path = "-".join(split_nodes[mlp_path_idx:])
        mlp_layer = int(split_nodes[mlp_path_idx][4:])
        device = self.hooked_model.device
        
        self.logger.info(f"Inspecting {complete_path}...")
        
        # Collect inputs and outputs for this MLP
        mlp_inputs = []
        mlp_outputs = []
        in_between_mlp = False  # Detects if this MLP belongs to a transformer head
        A_lis = []
        
        # For every node AFTER the MLP being inspected (note paths are reversed)...
        for node in split_nodes[:mlp_path_idx]:
            if node.startswith("mlp") and not in_between_mlp:
                in_between_mlp = True
            elif node.startswith("attn_output") and in_between_mlp:
                # If there is an attention head outputting to an MLP, we will need to verify that
                # the attention head weights are roughly one-hot, otherwise interpretation could
                # be misleading.
                #
                # Note: these need to be one-hot because otherwise they could represent a non-linear
                # transformation that is hard to interpret
                _, layer, head = node.split("-")
                layer, head = int(layer), int(head)
                A_lis.append((layer, head))
        
        # Collect relevant input and output pairs for target MLP
        mean_A_max = []
        pbar = tqdm(total=self.cache_num)
        for idx, batch in enumerate(self.dataloader):
            _ = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Collect the output for the MLP of interest
            mlp_out = None
            
            def temp_hook(module, input, output):
                nonlocal mlp_out
                mlp_out = self.hooked_model.activations[mlp_path]
            
            # Register hook to relevant MLP
            handle = self.hooked_model.model.transformer.h[mlp_layer].mlp.register_forward_hook(temp_hook)
            
            # Run inference and determine output using masks
            self.hooked_model(masks=torch.ones((1, 1), device=device), oa_vecs=self.oa_vecs, **batch)
            masks = (batch['input_ids'] != self.tokenizer.pad_token_id) & (batch['input_ids'] != self.tokenizer.eos_token_id)
            handle.remove()
            
            # Collect the relevant inputs by tracing them (i.e. transforming batch inputs up until target MLP)
            input_dependent = trace_mlp(
                self.hooked_model,
                self.converted_mlp,
                mlp_path,
                batch['input_ids'],
                batch['position_ids']
            )
            input_dependent = input_dependent[masks]
            mlp_out = mlp_out[masks]
            
            # Remove duplicates within a batch
            # Do so by using the distance between inputs as a measure of similarity
            distance = torch.cdist(input_dependent.unsqueeze(0), input_dependent.unsqueeze(0)).squeeze(0)
            selected_ids = (distance < 1e-3).float().argmax(dim=1).unique(sorted=False)
            mlp_inputs.append(input_dependent[selected_ids])
            mlp_outputs.append(mlp_out[selected_ids])
            
            # If there are downstream attention heads with downstream MLPs...
            if A_lis:
                
                # Compute the aggregated/absorbed weights for all downstream heads
                A_aggregated = None
                for layer, head in A_lis:
                    A = get_attn_weights_for_head(self.hooked_model, layer, head)
                    
                    if A_aggregated is None:
                        A_aggregated = A
                    else:
                        A_aggregated = A_aggregated @ A
                
                # Compute how "one-hot" the downstream heads are. If they are one-hot,
                # the maximum weight for each token should be approximately 1. Similarly,
                # the mean across all batches should be approximately 1
                mean_A_max.append(A_aggregated[masks].max(dim=-1)[0].mean())
            
            pbar.update(mlp_inputs[-1].size(0))
            if sum(item.size(0) for item in mlp_inputs) > self.cache_num:
                break
        
        pbar.close()

        # If there are downstream attention heads, they should be roughly one-hot (mean of maximum weights close to 1)
        if mean_A_max:
            mean_A_max = torch.stack(mean_A_max).mean().item()
            if mean_A_max < 0.9:
                self.logger.warning("RESULTS ARE NOT RELIABLE")
        
        # Collect inputs and outputs
        mlp_inputs = torch.cat(mlp_inputs, dim=0)
        mlp_outputs = torch.cat(mlp_outputs, dim=0)
        assert torch.allclose(mlp_inputs.sum(dim=1), torch.ones(mlp_inputs.size(0), device=device), atol=1e-3)
        
        # Trace the relevant MLP inputs downstream
        idx = mlp_path_idx - 1
        while idx >= 0:
            node = split_nodes[idx]
            
            if node == "lm_head":
                # If unembedding layer, apply LayerNorm and unembedding layer
                W_ln = get_ln_matrix_for_node(self.hooked_model.model, self.oa_vecs, None, "lm_head", None)
                mlp_outputs = mlp_outputs @ W_ln @ self.hooked_model.model.lm_head.weight.data.T
            elif node.startswith("attn_output"):
                # If attention head, apply LayerNorm, Value Weights and Output Weights
                _, layer, head = node.split("-")
                layer, head = int(layer), int(head)
                past_path = "-".join(split_nodes[idx + 1:])
                
                W_ln = get_ln_matrix_for_node(self.hooked_model.model, self.oa_vecs, layer, "v", head, past_path)
                W_v, W_o = get_ov_for_head(self.hooked_model.model, layer, head)
                mlp_outputs = mlp_outputs @ W_ln @ W_v @ W_o
            elif node.startswith("mlp"):
                # If MLP, apply linear LayerNorm and learned single-source MLP
                layer = int(node.split("-")[1])
                input_path = "-".join(split_nodes[idx+1:])
                
                ln_var = self.hooked_model.oa_vecs.ln_var[self.hooked_model.oa_vecs.to_ln_idx[(layer, "mlp", input_path)]].exp()
                input_activation = self.hooked_model._linear_layer_norm(self.hooked_model.ln_2[layer], mlp_outputs.unsqueeze(0), ln_var)
                
                mlp_outputs = self.hooked_model.oa_vecs.mlps[f'{layer} {input_path}'](input_activation).squeeze(0)
            else:
                raise RuntimeError(f"Invalid Node Encountered during LogitLens: {node}")
            
            idx -= 1

        return mlp_inputs, mlp_outputs
    
    @torch.no_grad()
    def _inspect_path(
        self,
        complete_path: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This function collects inputs and outputs for the MLP in a path when that path is potentially already symbolic.
        This can happen when analyzing the effect of an MLP on a select operator (QK vertex) as one side of the path
        could be fully symbolic whereas the other side could be the one containing the unexplained MLP. As such, this
        function generates input and output pairs needed for visualization when a path does not contain any
        unexplained MLPs

        Args:
            complete_path (str): the path of interest, with no unexplained MLPs
        """
        
        #TODO: eventually add a similar function for multi-source
        all_mlp_inputs = []
        all_mlp_outputs = []
        
        for i, batch in enumerate(self.dataloader):
            labels = batch.pop("labels")
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            self.hooked_model(masks=torch.ones((1, 1), device=self.device), oa_vecs=self.oa_vecs, **batch)
            
            # Collect the inputs and absorbed weights for a fully symbolic path
            dep_prod, indep_prod = get_input_weights_symbolic(
                self.hooked_model,
                self.oa_vecs,
                self.converted_mlp,
                complete_path,
                batch['input_ids'],
                batch['position_ids']
            )
            masks = (batch['input_ids'] != self.tokenizer.pad_token_id) & (batch['input_ids'] != self.tokenizer.eos_token_id)
            
            mlp_inputs = dep_prod[masks]
            mlp_outputs = mlp_inputs @ indep_prod
            
            # Remove duplicates within a batch
            dist = torch.cdist(mlp_inputs.unsqueeze(0), mlp_inputs.unsqueeze(0)).squeeze(0)
            selected_ids = (dist < 1e-3).float().argmax(dim=1).unique(sorted=False)
            all_mlp_inputs.append(mlp_inputs[selected_ids])
            all_mlp_outputs.append(mlp_outputs[selected_ids])
            
            if sum(item.size(0) for item in all_mlp_inputs) > 500:
                break
        
        all_mlp_inputs = torch.cat(all_mlp_inputs, dim=0)
        all_mlp_outputs = torch.cat(all_mlp_outputs, dim=0)
        
        assert torch.allclose(all_mlp_inputs.sum(dim=1), torch.ones(all_mlp_inputs.size(0), device=self.device), atol=1e-3)
        return all_mlp_inputs, all_mlp_outputs
        
    
    @torch.no_grad()
    def inspect_mlp_logits(self, complete_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This method selects a set of representative tokens that are most rewarded/suppressed
        by the MLP of interest. In particular, we produce around 100 examples of each class
        of token. The output of this function can be used to interpret the effect of the MLP
        on the output vocabulary

        Args:
            complete_path (str): the path ending in an unembedding layer

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: input, output representative pairs
        """
        #TODO: make this function handle both multi-source and single-source weight absorption
        
        assert complete_path.startswith("lm_head")
        mlp_in, mlp_logits = self._inspect_mlp(complete_path)
        if not hasattr(self.dataloader.dataset, "bce") or not self.dataloader.dataset.bce:
            # For algorithmic tasks, subtract the second largest value from each token to aid interpretation
            # Why?: because of softmax, only the relative differences in logits matter. By centering the data
            # in this way, we emphasize the specific token promoted the most by the MLP, while "zeroing out"
            # other (background) tokens
            mlp_logits = mlp_logits - mlp_logits.topk(dim=1, k=2)[0][:, 1:2]
        
        
        # Collect representative samples for visualization:
        # For each token, we identify the maximum positive logit value and divide this range
        # into 10 bins. We then identify the top 10 samples whose projected effect falls into each
        # magnitude range
        input_cache = []  # from highest to lowest
        output_cache = []
        
        num_bins = 20
        num_per_bin = 10
        max_v = mlp_logits.clamp(min=0).max(dim=0)[0]
        bin_edges = max_v.unsqueeze(0) * (torch.arange(1, num_bins // 2 + 1, dtype=torch.float, device=mlp_logits.device) / (num_bins // 2)).unsqueeze(1)
        
        for i in range(num_bins // 2 - 1, -1, -1):
            topk_indices = mlp_logits.masked_fill(mlp_logits > bin_edges[i:i+1], -100_000).topk(k=num_per_bin, dim=0, largest=True)[1]  # (k, vocab_size)
            input_cache.append(mlp_in[topk_indices].clone())  # (k, vocab, in_vocab)
            output_cache.append(mlp_logits[topk_indices].clone())  # (k, vocab, vocab)
        
        # Similar process to above, but identify inputs that cause the MLP to suppress specific output tokens most strongly
        min_v = mlp_logits.clamp(max=0).min(dim=0)[0]
        bin_edges = min_v.unsqueeze(0) * (torch.arange(1, num_bins // 2 + 1, dtype=torch.float, device=mlp_logits.device) / (num_bins // 2)).unsqueeze(1)
        
        for i in range(num_bins // 2):
            topk_indices = mlp_logits.masked_fill(mlp_logits < bin_edges[i:i+1], 100_000).topk(k=num_per_bin, dim=0, largest=False)[1]  # (k, vocab_size)
            input_cache.append(mlp_in[topk_indices].clone())  # (k, vocab, in_vocab)
            output_cache.append(mlp_logits[topk_indices].clone())  # (k, vocab, vocab)
        
        return input_cache, output_cache
    
    @torch.no_grad()
    def inspect_mlp_attn(
        self,
        q_complete_path: str,
        k_complete_path: str,
        attn_layer_idx: int,
        attn_head_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        This method collects representative queries, keys and attention logits in bins as before
        to aid interpretation of the MLP's impact on the attention head weights. In particular,
        we collect the top 10 most relevant queries and keys for each bin (computed as having the
        largest/smallest logits).

        Args:
            q_complete_path (str): the path ending in a query vertex
            k_complete_path (str): the path ending in a key vertex
            attn_layer_idx (int): the transformer layer of interest
            attn_head_idx (int): the attention head of interest

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: representative q_inputs, k_inputs and outputs for interpretation
        """
        
        self.logger.info(f"Inspecting: {(q_complete_path, k_complete_path)}")
        
        # Collect inputs and outputs for path ending in query
        q_mlp_inputs, q_mlp_outputs = self._inspect_mlp(q_complete_path)
        if q_mlp_inputs is None:
            # If no unexplained MLP in this path, we trace as normal
            q_mlp_inputs, q_mlp_outputs = self._inspect_path(q_complete_path)
        
        W_ln = get_ln_matrix_for_node(self.hooked_model.model, self.oa_vecs, attn_layer_idx, "q", attn_head_idx)
        W_q = get_qk_for_head(self.hooked_model.model, attn_layer_idx, "q", attn_head_idx)
        q_mlp_outputs = q_mlp_outputs @ W_ln @ W_q
        
        # Collect inputs and outputs for path ending in key
        k_mlp_inputs, k_mlp_outputs = self._inspect_mlp(k_complete_path)
        if k_mlp_inputs is None:
            # If no unexplained MLP in this path, we trace as normal
            k_mlp_inputs, k_mlp_outputs = self._inspect_path(k_complete_path)
        
        W_ln = get_ln_matrix_for_node(self.hooked_model.model, self.oa_vecs, attn_layer_idx, "k", attn_head_idx)
        W_k = get_qk_for_head(self.hooked_model.model, attn_layer_idx, "k", attn_head_idx)
        k_mlp_outputs = k_mlp_outputs @ W_ln @ W_k
        
        # Compute attention logits (and center to emphasize relative strengths)
        attn_logits = q_mlp_outputs @ k_mlp_outputs.T
        attn_logits -= attn_logits.mean(dim=1, keepdim=True)
        
        # Collect representative token positions for this MLP
        # We collect 10 positive logit interactions and 10 negative logit interactions as before
        num_bins = 20
        num_per_bin = 10
        num_q_cluster = 5
        
        q_input_cache = [[] for i in range(num_bins)]  # (k, num_q_cluster, in_vocab) -> highest to lowest
        k_input_cache = [[] for i in range(num_bins)]
        output_cache = [[] for i in range(num_bins)]
        
        # To reduce the number of interactions we need to inspect, we apply KMeans clustering (k=5) to the
        # Query vertex inputs. Within each cluster, we then find the highest and lowest attention logits
        # and bin the respective range to select the top 10 samples per bin as before
        kmeans = KMeans(n_clusters=num_q_cluster, random_state=0).fit(q_mlp_inputs.numpy(force=True))
        cluster_labels = torch.tensor(kmeans.labels_, device=self.device)
        
        for c_idx in range(num_q_cluster + 1):
            
            if c_idx == 0:
                selected_attn_logits = attn_logits
                selected_q_mlp_inputs = q_mlp_inputs
            else:
                selected_attn_logits = attn_logits[cluster_labels == c_idx - 1]
                selected_q_mlp_inputs = q_mlp_inputs[cluster_labels == c_idx - 1]
                if selected_attn_logits.numel() < num_per_bin:
                    continue
                
            
            max_v = selected_attn_logits.clamp(min=0).max()
            bin_edges = max_v * torch.arange(1, num_bins // 2 + 1, dtype=torch.float, device=self.device) / (num_bins // 2)
            
            # For all of the bins from 0 to the maximum positive logit, collect
            # 1. 10 queries with the largest contribution in this bin
            # 2. 10 keys with the largest contribution in this bin
            # 3. The 10 largest attention logits for this bin
            for i in range(num_bins // 2 - 1, -1, -1):
                topk_values, topk_indices = selected_attn_logits.masked_fill(selected_attn_logits > bin_edges[i], -100_000).view(-1).topk(k=num_per_bin, largest=True)
                q_indices = topk_indices // k_mlp_outputs.size(0)
                k_indices = topk_indices % k_mlp_outputs.size(0)
                
                q_input_cache[num_bins // 2 - 1 - i].append(selected_q_mlp_inputs[q_indices].clone())  # (k, in_vocab)
                k_input_cache[num_bins // 2 - 1 - i].append(k_mlp_inputs[k_indices].clone())  # (k, in_vocab)
                output_cache[num_bins // 2 - 1 - i].append(topk_values)  # (k,)
            
            min_v = selected_attn_logits.clamp(max=0).min()
            bin_edges = min_v * torch.arange(1, num_bins // 2 + 1, dtype=torch.float, device=self.device) / (num_bins // 2)
            
            # For all of the bins from minimum logit to 0, collect
            # 1. 10 queries with the smallest contribution in this bin
            # 2. 10 keys with the smallest contribution in this bin
            # 3. The 10 smallest attention logits for this bin
            for i in range(num_bins // 2):
                topk_values, topk_indices = selected_attn_logits.masked_fill(selected_attn_logits < bin_edges[i], 100_000).view(-1).topk(k=num_per_bin, largest=False)
                q_indices = topk_indices // k_mlp_outputs.size(0)
                k_indices = topk_indices % k_mlp_outputs.size(0)
                
                q_input_cache[num_bins // 2 + i].append(selected_q_mlp_inputs[q_indices].clone())  # (k, in_vocab)
                k_input_cache[num_bins // 2 + i].append(k_mlp_inputs[k_indices].clone())  # (k, in_vocab)
                output_cache[num_bins // 2 + i].append(topk_values)  # (k,)
        
        q_input_cache = [torch.stack(item, dim=1) for item in q_input_cache]
        k_input_cache = [torch.stack(item, dim=1) for item in k_input_cache]
        output_cache = [torch.stack(item, dim=1) for item in output_cache]
        
        return q_input_cache, k_input_cache, output_cache
