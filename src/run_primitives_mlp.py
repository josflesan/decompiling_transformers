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
    mlp_primitives_pipeline.run()
    
    # TEST: check to see if we can get primitive and run them
    # test_primitive = build_primitive(PrimitiveType.ZEROONE, pow=0.5, center=0.5)
    # out = test_primitive.apply(torch.rand(size=(3, 4)))
    
    # assert test_primitive.output_dim(torch.Size([3, 4])) == out.shape
    
    # print(f"The output tensor: {out}")

if __name__ == "__main__":
    main()