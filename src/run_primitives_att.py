import argparse
from transformers import logging

import primitives_att.primitives  # noqa: F401 — register primitives
from primitives_att.pipeline import AttPrimitivePipeline
from primitives_att.utilities.att_primitive_utils import load_config

def main():
    logging.set_verbosity_error()

    parser = argparse.ArgumentParser(description="Run Att Primitive Replacement")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config file",
    )
    args = parser.parse_args()
    run_config = load_config(args.config)

    pipeline = AttPrimitivePipeline(run_config)
    converted_att = pipeline.run()
    print(f"Attention primitive replacement complete. {len(converted_att)} layer entries saved.")

if __name__ == "__main__":
    main()
