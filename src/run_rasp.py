import argparse

from rasp.pipeline import RaspPipeline
from rasp.utilities.rasp_utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RASP decompilation")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config file",
    )
    args = parser.parse_args()
    run_config = load_config(args.config)

    pipeline = RaspPipeline(run_config)
    result = pipeline.run()
    print(f"RASP decompilation complete. Generated {len(result.lines)} program lines.")


if __name__ == "__main__":
    main()
