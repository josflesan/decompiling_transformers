import argparse

from transformers import logging

from decompilation.pipeline import DecompilationPipeline
from decompilation.utilities.decompilation_utils import load_config


def main() -> None:
    logging.set_verbosity_error()

    parser = argparse.ArgumentParser(
        description="Run the full decompilation pipeline (pruning → MLP → attention primitives)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the global decompilation config file",
    )
    parser.add_argument(
        "--skip-pruning",
        action="store_true",
        help="Skip causal pruning (requires existing pruning artifacts)",
    )
    parser.add_argument(
        "--skip-mlp-primitives",
        action="store_true",
        help="Skip MLP primitive replacement (requires existing MLP artifacts for attention stage)",
    )
    parser.add_argument(
        "--skip-att-primitives",
        action="store_true",
        help="Skip attention primitive replacement",
    )
    args = parser.parse_args()

    run_config = load_config(
        args.config,
        run_pruning=False if args.skip_pruning else None,
        run_mlp_primitives=False if args.skip_mlp_primitives else None,
        run_att_primitives=False if args.skip_att_primitives else None,
    )
    DecompilationPipeline(run_config).run()


if __name__ == "__main__":
    main()
