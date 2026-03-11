import argparse
import torch
import torch.nn.functional as F
import json
from copy import deepcopy
from collections import defaultdict
from dataclasses import asdict
from functools import partial
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel
from typing import Any, Dict, List, Tuple, Union

from data.CountDataset import CountDataset
from data.CustomCollator import CustomCollator
from pruning.core.hooks import GPT2ComponentHooks
from pruning.core.OptimalAblationVectors import OptimalAblationVectors
from pruning.core.mask_samplers import ComponentMaskSampler
from pruning.tasks.registry import get_task
from pruning.utilities.pruning_dataclasses import PruningRunConfig
from pruning.utilities.pruning_utils import load_config, output_model_arch_json, get_full_possible_config_for_pruning

@torch.no_grad()
def capture_ln_var(
    dataset: Union[CountDataset],
    model: GPT2LMHeadModel,
    collator: CustomCollator,
    device: torch.device,
    ignore_ids: List[int]
):
    """
    Estimate average input variance of every LayerNorm in the model.
    We do this by running 10 forward passes with the original model,
    each with batch size 64. The final variance of each input tensor
    is computed as the average of these runs

    Args:
        dataset (CountDataset | TODO): the dataset used for the task
        model (GPT2LMHeadModel): the original unpruned model
        collator (CustomCollator): the collator used to batch the task data
        device (torch.device): PyTorch device used for training
        ignore_ids (List[int]): token ids that should be ignored

    Returns:
        torch.Tensor: a tensor with the target variance for each of the input tensors to the model's LayerNorms
    """
    num_layers = len(model.transformer.h)
    batch_size = 64
    hooks = []
    var = torch.zeros(num_layers * 2 + 1, device=device)
    
    def save_hook(module, input, output, idx):
        """Accumulates the variance for this batch along the hidden dim. We mask out PAD/BOS/EOS/SEP tokens"""
        var[idx] += input[0].var(dim=-1)[~mask].mean()
    
    hooks.append(model.transformer.ln_f.register_forward_hook(partial(
        save_hook, idx=-1
    )))
    
    for layer in range(num_layers):
        hooks.append(model.transformer.h[layer].ln_1.register_forward_hook(partial(
            save_hook, idx=layer*2
        )))
        
        hooks.append(model.transformer.h[layer].ln_2.register_forward_hook(partial(
            save_hook, idx=layer*2+1
        )))
    
    inputs = []
    step_idx = 0
    for item in dataset:
        inputs.append(item)
        if len(inputs) == batch_size:
            batch = collator(inputs)
            batch = {k : v.to(device) for k, v in batch.items()}
            mask = torch.stack([batch['input_ids'] == i for i in ignore_ids]).any(dim=0)
            batch.pop("labels")
            model(**batch)
            
            inputs = []
            step_idx += 1
            if step_idx == 10:
                break
            
    for hook in hooks:
        hook.remove()
    
    var /= 10
    return var

def main(config_path: str):
    # Read configuration
    parser = argparse.ArgumentParser(description="Run pruning")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help='Path to the config file'
    )
    args = parser.parse_args()
    run_config: PruningRunConfig = load_config(args.config)
        
    #TODO: understand what this config is for
    output_config_path = run_config.full_output_dir / 'args.json'
    # with open(output_config_path, "w") as f:
    #     json.dump(asdict(run_config), f)
    output_dict = {}
    
    # Set up logging
    #TODO: set up logging
    
    # Load the model
    model = GPT2LMHeadModel.from_pretrained(Path(run_config.model_path)).to(run_config.torch_device)
    model.eval()
    
    original_model = GPT2LMHeadModel.from_pretrained(Path(run_config.model_path)).to(run_config.torch_device)
    original_model.eval()
    
    # Get the tokenizer and dataset from the task configuration
    task_config = run_config.task_config
    task = get_task(task_config.name, task_config)
    tokenizer, datasets = task.build()

    #TODO: log the tokenizer's vocabulary
    
    # Set up collator and dataloader
    collator = CustomCollator(tokenizer.pad_token_id)
    
    # Get the configuration for the model
    num_layers = len(model.transformer.h)
    num_heads_per_layer = {layer: model.transformer.h[layer].attn.num_heads for layer in range(num_layers)}
    model_config = get_full_possible_config_for_pruning(num_heads_per_layer)
    
    # Get Mask Sampler, Initialized Hooked Model and Trainable Parameters
    mask_sampler = ComponentMaskSampler(model_config)
    if run_config.init_sample_param:
        mask_sampler.sample_params.data *= run_config.init_sample_param
    
    linear_LN = run_config.stage_config.linear_ln
    hooked_model = GPT2ComponentHooks(model, model_config, mask_sampler.mapping_to_param_idx, linearLN=True)
    #TODO: log mask param names and total edge count
    
    # Compute the expected variance estimates of LayerNorm inputs
    converted_ln_var = torch.zeros(len(mask_sampler.output_vertex))
    if linear_LN:
        # Compute expected variance estimates
        ln_var = capture_ln_var(
            datasets["train"],
            original_model,
            collator,
            run_config.torch_device,
            [tokenizer.pad_token_id, tokenizer.eos_token_id]
        )
        
        # For each output nodes, map transformer graph vertices to the appropriate
        # LN variance estimate
        for i, output_v in enumerate(mask_sampler.output_vertex):
            match output_v:
                # LN1
                case (layer, act, head):
                    converted_ln_var[i] = ln_var[layer * 2]
                # LN2
                case (layer, mlp):
                    converted_ln_var[i] = ln_var[layer * 2 + 1]
                # LNF
                case (lm_head,):
                    converted_ln_var[i] = ln_var[-1]
            
        # Log the variance tensor
        converted_ln_var = converted_ln_var.log()
    else:
        #TODO: log this instead of printing
        print("USING REAL LAYERNORM")
    
    # Initialize the Optimal Ablation Vectors
    oa_vecs = OptimalAblationVectors(
        input_vertex=mask_sampler.input_vertex,
        output_vertex=mask_sampler.output_vertex,
        ln_vertex=mask_sampler.output_vertex,
        mlp_vertex=None,
        model_config=model.config,
        init_var=converted_ln_var
    ).to(run_config.torch_device)
    
    # Accumulate biases from previous attention layers, this gives a baseline
    # residual value entering this node and makes pruning less destructive
    # Without this, pruning might incorrectly keep some edges to recreate the missing
    # bias offset
    for layer in range(num_layers):
        for head in range(num_heads_per_layer[layer]):
            for act in ["q", "k", "v"]:
                for i in range(layer):
                    oa_vecs.output_vertex_oa.data[oa_vecs.to_out_oa_idx[(layer, act, head)]] += model.transformer.h[i].attn.resid_dropout(model.transformer.h[i].attn.c_proj.bias)
        
        for i in range(layer + 1):
            oa_vecs.output_vertex_oa.data[oa_vecs.to_out_oa_idx[(layer, "mlp")]] += model.transformer.h[i].attn.resid_dropout(model.transformer.h[i].attn.c_proj.bias)
    
    for i in range(num_layers):
        oa_vecs.output_vertex_oa.data[oa_vecs.to_out_oa_idx[("lm_head",)]] += model.transformer.h[i].attn.resid_dropout(model.transformer.h[i].attn.c_proj.bias)
    
    lamb = run_config.stage_config.lamb
    num_steps = run_config.stage_config.num_steps
    
    # If there are no eligible edges left to prune, save the model and end
    if mask_sampler.sample_params.numel() == 0:
        hooked_model.remove_hooks()
        # output_dict["result_patching_performance_global_iteration_0"] = {
        #     "acc_match": acc_match,
        #     "acc_task": acc_task,
        #     "task_loss": task_loss,
        #     "kl_div": kl_div,
        #     "num_edges": 0,
        #     "coef": lamb
        # }
        output_dict['result_patching_config_global_iteration_0'] = deepcopy(model_config)
    
    # Initialize optimizers, gamma and other training variables
    gamma = 0.0
    log_interval = 50
    batch_size = run_config.stage_config.batch_size
    num_repeat = run_config.stage_config.num_repeat
    unique_input_per_batch = batch_size // num_repeat
    
    assert unique_input_per_batch * num_repeat == batch_size
    
    sampling_opt = torch.optim.AdamW(
        mask_sampler.parameters(),
        lr=run_config.lr_sampler_for_pruning,
        weight_decay=0,
        betas=(0.9, 0.995)
    )
    oa_optimizer = torch.optim.AdamW(
        [
            {"params": [oa_vecs.ln_var], "lr": run_config.lr_ln_var_for_pruning},
            {"params": [oa_vecs.input_vertex_oa], "lr": run_config.lr_oa_for_pruning},
            {"params": [oa_vecs.output_vertex_oa], "lr": 1e-4},
        ],
        weight_decay=0
    )
    
    print("1. STARTING TRAINING LOOP")

    # Start training
    count_down = num_steps
    patience = 3
    training_logs = defaultdict(list)
    train_dataloader = DataLoader(datasets['train'], batch_size=unique_input_per_batch, shuffle=False, collate_fn=collator)
    for current_step, batch in enumerate(train_dataloader):
        # Move to device
        batch = {k: v.to(run_config.torch_device).repeat(num_repeat, *([1] * (v.dim() - 1))) for k, v in batch.items()}
        labels = batch.pop("labels")
        
        with torch.no_grad():
            target_logits = original_model(**batch).logits
            
        print("1.1 TARGET LOGITS COMPUTED")
        
        masks = mask_sampler.sample_masks(batch_size)
        logits = hooked_model(masks=masks, oa_vecs=oa_vecs, **batch).logits
        
        print("1.2 HOOKED LOGITS COMPUTED")
        
        #TODO: compute the task loss and the pruning loss
        task_loss = F.cross_entropy(logits[:, :-1].flatten(end_dim=1), labels[:, 1:].flatten()).item()
        target_logits = target_logits[:, :-1][labels[:, 1:] != -100]
        logits = logits[:, :-1][labels[:, 1:] != -100]
        loss = F.kl_div(
            F.log_softmax(logits, dim=-1), F.log_softmax(target_logits, dim=-1),
            log_target=True
        )
        
        training_logs['kl_div'].append(loss.item())
        training_logs['task_loss'].append(task_loss)
        penalty, (reg_edge, reg_node) = mask_sampler.get_penalty(gamma)
        training_logs['reg_edge'].append(reg_edge)
        training_logs['reg_node'].append(reg_node)        
        loss = loss + lamb * penalty
        
        sampling_opt.zero_grad()
        oa_optimizer.zero_grad()
        loss.backward()
        
        print("1.3 BACKPROP SUCCESSFUL")
        
        # Clip gradient norms and take step
        training_logs["oa_grad_norm"].append(torch.nn.utils.clip_grad_norm_(oa_vecs.parameters(), max_norm=float('inf')).item())
        training_logs["sampler_grad_norm"].append(torch.nn.utils.clip_grad_norm_(mask_sampler.parameters(), max_norm=float('inf')).item())
        torch.nn.utils.clip_grad_norm_(mask_sampler.parameters(), 5)
        sampling_opt.step()
        oa_optimizer.step()
        
        print("1.4 TRAINING LOGS")
        print(training_logs)
        
        # Log warning if any mask parameters are NaN
        nan_count = sum(p.isnan().sum().item() for p in mask_sampler.parameters())
        if nan_count > 0:
            print("WARNING: sum NaN ", nan_count)
        
        # Printing
        if (current_step + 1) % log_interval == 0:
            #TODO: log the average training loss in the logs
            all_sample_p = torch.cat([p.data.detach().view(-1) for p in mask_sampler.parameters()], dim=0)
            hist, bin_edges = torch.histogram(all_sample_p.cpu(), bins=5)
            #TODO: log histogram of sampling parameters
            
            if all_sample_p.max().item() < -2:
                print("All pruned, training is failed. Stop early...")
                break
            
            if run_config.baseline_loss and (sum(training_logs['kl_div']) / len(training_logs['kl_div'])) > run_config.baseline_loss:
                patience -= 1
                if patience == 0:
                    print("Loss stuck at high value, training is failed. Stop early...")
                    break
            else:
                patience = 3
            
            if ((all_sample_p > -1) & (all_sample_p < 1)).sum().item():
                count_down = num_steps + 1
            
            training_logs = defaultdict(list)
        
        # Early stopping
        count_down -= 1
        if count_down == 0:
            break
        if (current_step + 1) == 5000:
            break
        
        break
    
    print("2. TRAINING COMPLETE!")
    
    # Start validation
    num_test_step = 200
    num_correct = 0
    num_match = 0
    task_loss = 0
    kl_div = 0
    dataloader = DataLoader(datasets['val'], batch_size=batch_size, shuffle=False, collate_fn=collator)
    loss_func = torch.nn.CrossEntropyLoss()
    
    print("3. STARTED VALIDATION")
    
    with torch.no_grad():
        for current_step, batch in enumerate(dataloader):
            # Move batch tensors to device
            batch = {k: v.to(run_config.torch_device) for k, v in batch.items()}
            labels = batch.pop("labels")
            
            target_logits = original_model(**batch).logits
            
            masks = mask_sampler.sample_binary_masks(batch_size)
            logits = hooked_model(masks=masks, oa_vecs=oa_vecs, **batch).logits
            
            shift_target_logits = target_logits[:, :-1]
            shift_logits = logits[:, :-1]
            shift_labels = labels[:, 1:]
            target_predictions = shift_target_logits.argmax(dim=-1)
            predictions = shift_logits.argmax(dim=-1)
            
            match = ((predictions == target_predictions) | (shift_labels == -100)).all(dim=1)
            num_match += match.sum().item()
            
            correct = ((predictions == shift_labels) | (shift_labels == -100)).all(dim=1)
            num_correct += correct.sum().item()
            
            task_loss += loss_func(shift_logits.flatten(end_dim=1), shift_labels.flatten()).item()
            kl_div += F.kl_div(F.log_softmax(shift_logits[shift_labels != -100], dim=-1), F.log_softmax(shift_target_logits[shift_labels != -100], dim=-1), log_target=True).item()
            
            
            if current_step + 1 == num_test_step:
                break
            break    
    
    acc_match = num_match / (num_test_step * batch_size)
    acc_task = num_correct / (num_test_step * batch_size)
    task_loss /= num_test_step
    kl_div /= num_test_step
    
    print("3. FINISHED VALIDATION")
    
    masks = mask_sampler.sample_binary_masks(1).squeeze(0)
    num_edges = (masks == 1).sum().item()
    print(f"After Pruning Edge Count: {num_edges}")
    # convert_mask_to_config_(masks, model_config, mask_sampler.mapping_to_param_idx)

    # Prepare output dictionary
    output_dict[f"result_patching_performance_global_iteration_0"] = {
        "acc_match": acc_match,
        "acc_task": acc_task,
        "task_loss": task_loss,
        "kl_div": kl_div,
        "num_edges": num_edges,
        "coef": lamb
    }
    
    #TODO: log end of global iteration
    
    output_dict[f"result_patching_config_global_iteration_0"] = deepcopy(model_config)
    
    hooked_model.remove_hooks()
    
    # Save optimal ablation vectors and output training results
    torch.save(oa_vecs, run_config.full_output_dir / 'oa_vecs.pt')
    output_dict['acc_match'] = acc_match
    output_dict['acc_task'] = acc_task
    output_dict['kl_div'] = kl_div
    output_dict['task_loss'] = task_loss
    
    output_file = run_config.full_output_dir / 'output.json'
    with open(output_file, "w") as f:
        json.dump(output_dict, f, indent=4)
    
    
    print("4. OA VECTORS SAVED, OUTPUT FILE CREATED")
    #TODO: cleanup the logger

if __name__ == "__main__":
    main('src/pruning/configs/prune_test.yaml')