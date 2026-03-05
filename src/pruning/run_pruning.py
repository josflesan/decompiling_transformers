import argparse
from pathlib import Path
from transformers import GPT2LMHeadModel

from core.hooks import GPT2ComponentHooks
from utilities.pruning_dataclasses import PruningRunConfig
from utilities.pruning_utils import load_config, output_model_arch_json, get_full_possible_config_for_pruning

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
    # output_config_path = output_dir / 'args.json'
    # with open(output_config_path, "w") as f:
    #     json.dump(config_dict, f)
    # output_dict = {}
    
    # Set up logging
    #TODO: set up logging
    
    # Load the model
    model = GPT2LMHeadModel.from_pretrained(Path(run_config.model_path)).to(run_config.torch_device)
    model.eval()
    
    # Pretty-print the model block config
    output_model_arch_json(model=model, out_dir=run_config.full_output_dir)
    
    #TODO: get tokenizer and dataset for this task
    #TODO: set up loss for the task
    
    # Get the configuration for the model
    num_layers = len(model.transformer.h)
    num_heads_per_layer = {layer: model.transformer.h[layer].attn.num_heads for layer in range(num_layers)}
    model_config = get_full_possible_config_for_pruning(num_heads_per_layer)
    
    print(model_config)
    
    #TODO: Finish pruning training section (MaskSampler, OptimalAblationVectors, etc.)
    # Try to instantiate model with hooks
    model_with_hooks = GPT2ComponentHooks(model, model_config, None)
    
    #TODO: implement training for stage 1

if __name__ == "__main__":
    main('src/pruning/configs/prune_test.yaml')