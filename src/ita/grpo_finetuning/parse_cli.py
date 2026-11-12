import argparse


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ArgLLMs GRPO Training")

    # ------------------------------------------------------------------ #
    # Config shortcut – if provided all other flags are ignored
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a YAML config file; all other flags are ignored when set.",
    )

    # ------------------------------------------------------------------ #
    # General
    # ------------------------------------------------------------------ #
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for the argument-generation model")
    parser.add_argument("--dataset", required=False, type=str, default=None,
                        help="Path to the CSV dataset")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--output-dir", type=str, default="outputs/grpo/Qwen3-8B-GRPO",
                        help="Output directory (relative to the repo root); its name is the HF repo name")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num_generatons", type=int, default=2,
                        help="Number of GRPO generations per prompt")
    parser.add_argument("--method", type=str, default="transformers")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None,
                        help="Path to a trainer checkpoint to resume training from")

    # ------------------------------------------------------------------ #
    # Semantics
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--semantics", type=str, required=False, default=None,
        choices=["dfq", "qe"],
        help="Gradual semantics to use for computing claim strength",
    )
    parser.add_argument(
        "--conservativeness", type=float, default=1.0,
        help="Conservativeness parameter for dfq / qe semantics",
    )
    parser.add_argument(
        "--upper-threshold", type=float, required=False, default=None,
        help="Predicted strength ≥ this → TRUE",
    )
    parser.add_argument(
        "--lower-threshold", type=float, required=False, default=None,
        help="Predicted strength ≤ this → FALSE",
    )

    # ------------------------------------------------------------------ #
    # Online threshold re-optimization (opt-in; off by default)
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--enable-threshold-optimization", action="store_true", default=False,
        help=(
            "Periodically re-optimize the reward's upper/lower classification "
            "thresholds from a recent window of generated completions. "
            "Disabled by default — static --upper-threshold/--lower-threshold "
            "are used unless this is set."
        ),
    )
    parser.add_argument(
        "--threshold-opt-window", type=int, default=50,
        help="Number of most recent training steps' completions to use for re-optimization.",
    )
    parser.add_argument(
        "--threshold-opt-interval", type=int, default=50,
        help="Re-optimize thresholds every N training steps.",
    )
    parser.add_argument(
        "--threshold-opt-min-samples", type=int, default=20,
        help="Minimum number of valid (non-parse-error) completions required before re-optimizing.",
    )
    parser.add_argument(
        "--threshold-opt-criterion", type=str, default="balanced_accuracy",
        choices=["balanced_accuracy", "accuracy"],
        help="Metric to maximize when selecting the new threshold delta.",
    )

    # ------------------------------------------------------------------ #
    # W&B
    # ------------------------------------------------------------------ #
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable Weights & Biases logging")
    parser.add_argument("--wandb-run-id", type=str, default=None,
                        help="Existing W&B run ID to resume logging into")
    parser.add_argument("--push-to-hub", action="store_true", default=False,
                        help="Upload checkpoints and artefacts to the HF Hub as <user>/<output dir name>")
    parser.add_argument("--continuous-reward", action="store_true", default=False,
                        help="Use continuous (smooth) reward instead of binary +100/-100")

    # ------------------------------------------------------------------ #
    # Base-score strategy subcommands (not required when --config is used)
    # ------------------------------------------------------------------ #
    parser = add_bs_parser(parser, required=False)

    return parser.parse_args()


def add_bs_parser(
    parser: argparse.ArgumentParser, required: bool = True
) -> argparse.ArgumentParser:
    """Add base-score strategy arguments to an existing parser."""

    parser.add_argument("--bs-device", type=str, default="cuda:0",
                        help="Device for the base-score model")

    subparsers = parser.add_subparsers(
        dest="base_score_strategy",
        required=required,
        help="Base-score strategy: constant_base_score | llm_base_score | learned_base_score",
    )

    # -- constant -------------------------------------------------------
    constant_p = subparsers.add_parser(
        "constant_base_score",
        help="Use a fixed scalar as the base score for all arguments",
    )
    constant_p.add_argument("--base-score", type=float, default=0.5,
                             help="The constant base score value (default: 0.5)")

    # -- LLM ------------------------------------------------------------
    subparsers.add_parser(
        "llm_base_score",
        help="Use an LLM to compute base scores",
    )

    # -- Learned (LLMRegressionModel / safetensors checkpoint) ------------
    learned_p = subparsers.add_parser(
        "learned_base_score",
        help="Use LLMRegressionModel (safetensors checkpoint) to compute base scores",
    )
    learned_p.add_argument("--bs-model-path", type=str, required=True,
                            help="Path to the LLMRegressionModel checkpoint directory")

    return parser