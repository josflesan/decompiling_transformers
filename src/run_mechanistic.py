import os
import warnings
from transformer_lens.model_bridge import TransformerBridge

from data.CountDataset import CountCorruption
from mechanistic.core.attribution import (
    residual_stream_attribution,
    layerwise_attribution,
    attention_head_attribution
)
from mechanistic.core.path_patching import path_patch
from mechanistic.utilities.mechinterp_utils import attn_z, attn_v, mlp_out
from mechanistic.utilities.metrics import logit_diff_metric
from mechanistic.utilities.mechinterp_dataclasses import CircuitNode
from mechanistic.utilities.mechinterp_viz import plot_path_patching_aggregated
from tasks.registry import get_task
from utilities.core import TaskConfig

os.environ['TRANSFORMERLENS_ALLOW_MPS'] = '1'
warnings.filterwarnings('ignore')

EXP_NAME = 'prune_test'
MODEL_PATH = "models/count-%2l4h256d4lr01drop"

def run_attribution_experiments(model, clean_corrupt_data):
    residual_stream_attribution(
        model=model,
        tokens=clean_corrupt_data.clean_tokens,
        position_ids=clean_corrupt_data.clean_pos,
        answer_tokens=clean_corrupt_data.answer_tokens[:, 0],
        exp_name=EXP_NAME
    )
    
    layerwise_attribution(
        model=model,
        tokens=clean_corrupt_data.clean_tokens,
        position_ids=clean_corrupt_data.clean_pos,
        answer_tokens=clean_corrupt_data.answer_tokens[:, 0],
        exp_name=EXP_NAME
    )
    
    attention_head_attribution(
        model=model,
        tokens=clean_corrupt_data.clean_tokens,
        position_ids=clean_corrupt_data.clean_pos,
        answer_tokens=clean_corrupt_data.answer_tokens[:, 0],
        exp_name=EXP_NAME
    )

def run_path_patching_experiments(model, clean_corrupt_data, exp=1):
    
    match exp:
        
        case 1:
            # CASE 1: Node->Head (as above)
            sender_nodes = {
                f"{attn_z(l)}.{h}": CircuitNode(name=attn_z(l), layer_idx=l, head_idx=h)
                for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)
            }
            sender_nodes.update({
                mlp_out(l): CircuitNode(name=mlp_out(l), layer_idx=l)
                for l in range(model.cfg.n_layers)
            })

            target_head = (1, 3)
            receiver_nodes = {
                attn_v(1): CircuitNode(name=attn_v(1), layer_idx=target_head[0], head_idx=target_head[1]),
            }

            title = f"Path Patching Results for Node->H{target_head[1]}.{target_head[0]}"
        
        case 2:
            # CASE 2: Head 0.0->[H1.0-3, MLP1]
            sender_nodes = {
                f"{attn_z(0)}.0": CircuitNode(name=attn_z(0), layer_idx=0, head_idx=0)
            }

            receiver_nodes = {
                f"{attn_z(1)}.0": CircuitNode(name=attn_z(1), layer_idx=1, head_idx=0),
                f"{attn_z(1)}.1": CircuitNode(name=attn_z(1), layer_idx=1, head_idx=1),
                f"{attn_z(1)}.2": CircuitNode(name=attn_z(1), layer_idx=1, head_idx=2),
                f"{attn_z(1)}.3": CircuitNode(name=attn_z(1), layer_idx=1, head_idx=3),
                mlp_out(1): CircuitNode(name=mlp_out(1), layer_idx=1)
            }
            
            title = f"Path Patching Results for H0.0->[H1.0-3, MLP1]"
    
    results = path_patch(
        model=model,
        sender_nodes=sender_nodes,
        receiver_nodes=receiver_nodes,
        clean_corrupt_data=clean_corrupt_data,
        metric=logit_diff_metric,
    )

    fig = plot_path_patching_aggregated(results, receiver_nodes, model, title=title)
    fig.show()

def main():
    # Test function to verify that the functions in core are working
    model = TransformerBridge.boot_transformers(
        MODEL_PATH,
        device='mps',
    )
    model.eval()
    
    # Get tokenizer and corrupted dataset
    config = TaskConfig(
        name='counting',
        train_length_range=[50, 150],
        val_length_range=[50, 150],
        max_test_length=150
    )
    task = get_task('counting', config)
    tokenizer, dataset = task.build()
    dataset = dataset['train']

    corrupted_data = dataset.get_corrupted(
        CountCorruption.CHANGE_START,
        batch_size=128
    )
    
    # Run attribution experiments
    run_attribution_experiments(model, corrupted_data)
    
    # Test path-patching function
    run_path_patching_experiments(model, corrupted_data, exp=2)


if __name__ == "__main__":
    main()