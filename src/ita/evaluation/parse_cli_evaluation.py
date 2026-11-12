import argparse

from ita.grpo_finetuning.parse_cli import add_bs_parser


def parse_evaluation_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate generated arguments with base score models"
    )

    parser.add_argument(
        "--arguments_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="./eval_results.csv",
    )

    parser.add_argument(
        "--semantics",
        type=str,
        required=True,
        choices=["dfq", "qe"],
    )

    parser.add_argument(
        "--conservativeness",
        type=float,
        required=False,
        default=1.0,
    )

    parser.add_argument(
        "--upper_threshold",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--lower_threshold",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--no_wandb",
        action="store_true",
    )

    parser = add_bs_parser(parser)

    return parser.parse_args()
