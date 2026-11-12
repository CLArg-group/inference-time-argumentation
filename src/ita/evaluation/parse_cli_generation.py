import argparse


def parse_generation_arguments():
    parser = argparse.ArgumentParser(
        description="Generate arguments for claims using LLM"
    )

    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen3-8B",
    )

    parser.add_argument(
        "--eval_data_paths",
        type=str,
        nargs="+",
        required=True,
        help="One or more paths to CSV files containing claims to generate arguments for",
    )

    parser.add_argument(
        "--label_column",
        type=str,
        default="valid",
        help="Name of the label column in the CSV files (default: 'valid'). Use 'auto' to auto-detect.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="./generated_arguments.json",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )

    parser.add_argument(
        "--no_wandb",
        action="store_true",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of claims to generate arguments for in parallel (default: 8)",
    )

    return parser.parse_args()
