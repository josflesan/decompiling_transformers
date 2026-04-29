import argparse
import torch

from primitives_mlp.pipeline import MLPPrimitivePipeline
from primitives_mlp.utilities.mlp_primitive_utils import load_config
from primitives_mlp.utilities.mlp_primitive_dataclasses import MLPPrimitivesConfig

def main():
    # Read configuration
    parser = argparse.ArgumentParser(description="Run MLP Primitive Replacement")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help='Path to the config file'
    )
    args = parser.parse_args()
    run_config: MLPPrimitivesConfig = load_config(args.config)
    
    # Initialize MLP Primitive Replacement Pipeline and run
    mlp_primitives_pipeline = MLPPrimitivePipeline(run_config)
    converted_mlp = mlp_primitives_pipeline.run()

if __name__ == "__main__":
    main()