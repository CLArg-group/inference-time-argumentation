"""Score arguments using a trained regression model checkpoint.

Reads arguments from a CSV file with columns: claim,is_support,argument
Outputs scores to a CSV file with an additional 'score' column.
"""

import argparse
import csv
from pathlib import Path
from typing import NamedTuple

from tqdm import tqdm

from ita.base_score_model.llm_regression_model import LLMRegressionModel
from ita.base_score_model.prime_regression_head import _build_conversation


class ArgumentRow(NamedTuple):
    """Represents a single argument from the input CSV."""

    claim: str
    is_support: bool
    argument: str


def load_arguments_from_csv(csv_path: str) -> list[ArgumentRow]:
    """Load arguments from a CSV file.

    Args:
        csv_path: Path to CSV file with columns: claim,is_support,argument

    Returns:
        List of ArgumentRow objects.
    """
    arguments: list[ArgumentRow] = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse is_support as boolean
            is_support_str = row["is_support"].strip().lower()
            if is_support_str in ("true", "1", "yes", "support"):
                is_support = True
            elif is_support_str in ("false", "0", "no", "attack"):
                is_support = False
            else:
                raise ValueError(
                    f"Invalid is_support value: {row['is_support']}. "
                    "Must be one of: true/false, 1/0, yes/no, support/attack"
                )

            arguments.append(
                ArgumentRow(
                    claim=row["claim"].strip(),
                    is_support=is_support,
                    argument=row["argument"].strip(),
                )
            )

    return arguments


def score_arguments(
    model: LLMRegressionModel,
    arguments: list[ArgumentRow],
    batch_size: int = 8,
) -> list[float]:
    """Score a list of arguments using the regression model.

    Args:
        model: Trained LLMRegressionModel model.
        arguments: List of ArgumentRow objects to score.
        batch_size: Number of arguments to process at once.

    Returns:
        List of float scores in [0, 1], one per argument.
    """
    model.eval()

    # Build conversations for all arguments
    conversations = [
        _build_conversation(arg.claim, arg.argument, arg.is_support)
        for arg in arguments
    ]

    # Score in batches to show progress
    all_scores: list[float] = []
    for i in tqdm(range(0, len(conversations), batch_size), desc="Scoring arguments"):
        batch = conversations[i : i + batch_size]
        scores = model.predict(batch, disable_tqdm=True)
        all_scores.extend(scores)

    return all_scores


def save_scores_to_csv(
    arguments: list[ArgumentRow],
    scores: list[float],
    output_path: str,
) -> None:
    """Save arguments and their scores to a CSV file.

    Args:
        arguments: Original arguments.
        scores: Predicted scores for each argument.
        output_path: Path to output CSV file.
    """
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["claim", "is_support", "argument", "score"])

        for arg, score in zip(arguments, scores):
            writer.writerow([arg.claim, arg.is_support, arg.argument, f"{score:.6f}"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score arguments using a trained regression model checkpoint."
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Directory of the trained regression head checkpoint to load.",
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Path to input CSV file with columns: claim,is_support,argument",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Path to output CSV file (will include scores).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for scoring (default: 8).",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device map for model loading (e.g. 'auto', 'cuda:0').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    # Validate paths
    if not Path(args.input_csv).exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    # Load arguments from CSV
    print(f"Loading arguments from {args.input_csv}...")
    arguments = load_arguments_from_csv(args.input_csv)
    print(f"Loaded {len(arguments)} arguments.")

    # Load model
    print(f"Loading model from {args.checkpoint_dir}...")
    model = LLMRegressionModel.load_checkpoint(
        args.checkpoint_dir,
        device_map=args.device_map,
        seed=args.seed,
    )
    print("Model loaded successfully.")

    # Score arguments
    print("Scoring arguments...")
    scores = score_arguments(model, arguments, batch_size=args.batch_size)

    # Save results
    print(f"Saving results to {args.output_csv}...")
    save_scores_to_csv(arguments, scores, args.output_csv)
    print("Done!")

    # Print some statistics
    print("\nScore statistics:")
    print(f"  Min:  {min(scores):.6f}")
    print(f"  Max:  {max(scores):.6f}")
    print(f"  Mean: {sum(scores) / len(scores):.6f}")


if __name__ == "__main__":
    main()
