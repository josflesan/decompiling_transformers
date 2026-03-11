import argparse

from pruning.pipeline import CausalPruningPipeline
from pruning.utilities.pruning_dataclasses import PruningRunConfig
from pruning.utilities.pruning_utils import load_config

def main():
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
    
    # Initialize pruning stage object and run
    pruning_pipeline = CausalPruningPipeline(run_config)
    pruning_pipeline.run()

    #TODO: cleanup the logger

if __name__ == "__main__":
    main()