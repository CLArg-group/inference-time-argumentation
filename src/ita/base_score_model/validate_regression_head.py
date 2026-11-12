"""Evaluate a trained LLMRegressionModel checkpoint on multiple validation datasets."""

import argparse
import csv
import math
import random
from pathlib import Path
from typing import Any, Protocol, override

import numpy as np
from scipy.stats import spearmanr
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, classification_report
from tqdm import tqdm
import wandb

from ita.base_score_model.data_utils import (
    ConsistencySample,
    RankingSample,
    load_consistency_samples,
    load_ranking_samples,
)
from ita.base_score_model.llm_regression_model import LLMRegressionModel
from ita.base_score_model.prime_regression_head import (
    _build_conversation,
    _parse_likelihood,
)
from ita.base_score_model.utils import parse_semantics
from ita.semantics import GradualSemantics
from ita.base_score_model.train_regression_head import (
    CLAIM_IDX,
    _build_adjacency_matrix,
    _CLASS_NAMES,
    _ternary_classify,
    average_metrics,
    hallucination_rate,
    load_training_samples,
    TrainingSample,
)


class Scorer(Protocol):
    """Protocol for scoring argument conversations.

    Maps a list of conversations to a list of float scores in [0, 1].
    """

    def __call__(self, conversations: list[list[dict[str, str]]]) -> list[float]:
        """Score a batch of conversations.

        Args:
            conversations: List of conversations, each formatted as a list of
                {"role": ..., "content": ...} dicts.

        Returns:
            List of float scores in [0, 1], one per conversation.
        """
        ...


class RegressionScorer(Scorer):
    """Scores arguments using the trained regression head."""

    def __init__(self, model: LLMRegressionModel):
        self.model = model

    @override
    def __call__(self, conversations: list[list[dict[str, str]]]) -> list[float]:
        self.model.eval()
        return self.model.predict(conversations, disable_tqdm=True)


class LLMScorer(Scorer):
    """Scores arguments using the base LLM's generated text (no regression head)."""

    def __init__(
        self, model: LLMRegressionModel, batch_size: int = 8, seed: int = 42
    ):
        self.model = model
        self.batch_size = batch_size
        self.seed = seed

    @override
    def __call__(self, conversations: list[list[dict[str, str]]]) -> list[float]:
        self.model.eval()
        raw = self.model.generate(
            input_conversations=conversations,
            batch_size=self.batch_size,
            seed=self.seed,
            disable_tqdm=True,
        )
        scores: list[float] = []
        for text in raw:
            value = _parse_likelihood(text)
            if value is None:
                raise ValueError(
                    f"Failed to parse LLM output '{text}' as a likelihood score."
                )
            scores.append(value)
        return scores


class ConstantScorer(Scorer):
    """Scores all arguments with a constant base score."""

    def __init__(self, score: float = 0.5):
        self.score = score

    @override
    def __call__(self, conversations: list[list[dict[str, str]]]) -> list[float]:
        return [self.score] * len(conversations)


def predict_with_scorer(
    samples: list[TrainingSample],
    semantics: GradualSemantics,
    score_fn: Scorer,
    estimate_claim_score: bool = False,
) -> tuple[list[float | None], list[dict[str, Any]]]:
    """Compute predictions using a custom scoring function.

    Mirrors :func:`train_regression_head.predict` but substitutes the
    regression-head forward pass with ``score_fn``, which maps a batch of
    conversations to scores in [0, 1].  This supports alternative scoring
    strategies such as base-LLM generation or constant 0.5 baselines.

    Args:
        samples (list[TrainingSample]): Evaluation samples.
        semantics (GradualSemantics): Argumentation semantics to apply.
        score_fn (Scorer): Function mapping a list of conversations to a
            list of float scores in [0, 1].
        estimate_claim_score (bool): If True, also pass the claim through
            ``score_fn`` using the analyst topic prompt instead of fixing its
            base score to 0.5.

    Returns:
        tuple[list[float | None], list[dict[str, Any]]]: Tuple of
            (predicted claim strengths, prediction data dictionaries). None values
            in the predictions list indicate samples that were skipped due to
            having no support or attack arguments. The data dictionaries contain
            metadata for each sample.
    """
    all_preds: list[float | None] = []
    all_data: list[dict[str, Any]] = []

    # Determine if we should use the base LLM prompt format
    use_base_llm_prompt = isinstance(score_fn, LLMScorer)

    for sample in tqdm(samples, desc="Evaluating"):
        if not sample.support and not sample.attack:
            all_preds.append(None)
            all_data.append({})
            continue

        support_convs = [
            _build_conversation(
                sample.claim, arg, True, use_base_llm_prompt=use_base_llm_prompt
            )
            for arg in sample.support
        ]
        attack_convs = [
            _build_conversation(
                sample.claim, arg, False, use_base_llm_prompt=use_base_llm_prompt
            )
            for arg in sample.attack
        ]
        raw_arg_scores = score_fn(support_convs + attack_convs)
        arg_scores = torch.tensor(raw_arg_scores, dtype=torch.float32)

        if estimate_claim_score:
            claim_score = torch.tensor(
                score_fn(
                    [
                        _build_conversation(
                            sample.claim,
                            topic=True,
                            use_base_llm_prompt=use_base_llm_prompt,
                        )
                    ]
                ),
                dtype=torch.float32,
            )
        else:
            claim_score = torch.tensor([0.5], dtype=torch.float32)

        base_scores = torch.cat([claim_score, arg_scores])
        A = _build_adjacency_matrix(len(sample.support), len(sample.attack))
        with torch.no_grad():
            final_strengths = semantics.forward(A, base_scores, base_scores)

        claim_strength_val = final_strengths[CLAIM_IDX].item()
        n_support = len(sample.support)
        support_base_scores = [round(s, 2) for s in raw_arg_scores[:n_support]]
        attack_base_scores = [round(s, 2) for s in raw_arg_scores[n_support:]]
        all_preds.append(claim_strength_val)
        all_data.append(
            {
                "claim": sample.claim,
                "claim_strength": f"{claim_strength_val:.6f}",
                "claim_base_score": claim_score.item(),
                "support_base_scores": str(support_base_scores),
                "attack_base_scores": str(attack_base_scores),
                "num_support": n_support,
                "num_attack": len(sample.attack),
                "qwen_argument_score_fallbacks": 0,
            }
        )

    return all_preds, all_data


# TODO: Ugly duplication of the evaluation code from train_regression_head.py.
def evaluate_with_scorer(
    samples: list[TrainingSample],
    semantics: GradualSemantics,
    score_fn: Scorer,
    estimate_claim_score: bool = False,
    scores: list[float | None] | None = None,
    prediction_metadata: list[dict[str, Any]] | None = None,
    delta: float | None = None,
    csv_output_path: Path | str | None = None,
) -> dict[str, float]:
    """Compute val metrics, optionally using the provided scores and prediction metadata.

    Args:
        samples (list[TrainingSample]): Evaluation samples.
        semantics (GradualSemantics): Argumentation semantics to apply.
        score_fn (Scorer): Function mapping a list of conversations to a
            list of float scores in [0, 1].
        estimate_claim_score (bool): If True, also pass the claim through
            ``score_fn`` using the analyst topic prompt instead of fixing its
            base score to 0.5.
        prediction_metadata (list[dict[str, Any]] | None): Metadata for each sample,
            including claim, base scores, and number of support/attack arguments.
            If None, it will be computed using ``predict_with_scorer``.
        scores (list[float | None] | None): Predicted or target scores.
            If None, it will be computed using ``predict_with_scorer``.
        delta (float | None): Parameter specifying the width of the "uncertain"
            class.  If None, the default boundaries of 1/3 and 2/3 are used.
        csv_output_path (Path | str | None): If provided, write a per-sample
            prediction CSV to this path in the same format as tfu_predictions.csv.

    Returns:
        (dict[str, float]): Same metrics as :func:`train_regression_head.evaluate`.
    """
    if scores is None or prediction_metadata is None:
        scores, prediction_metadata = predict_with_scorer(
            samples, semantics, score_fn, estimate_claim_score=estimate_claim_score
        )

    all_preds: list[float] = []
    all_targets: list[float] = []
    csv_rows: list[dict] = []

    for sample, score, metadata in tqdm(
        list(zip(samples, scores, prediction_metadata, strict=True)), desc="Evaluating"
    ):
        if score is None:
            continue

        claim_strength_val = score
        n_support = len(sample.support)
        gold_cls = _ternary_classify([sample.label], delta=delta)[0].capitalize()
        pred_cls = _ternary_classify([claim_strength_val], delta=delta)[0].capitalize()
        csv_rows.append(
            {
                "row_idx": len(csv_rows),
                "claim": sample.claim,
                "gold_label": gold_cls,
                "prediction": pred_cls,
                "is_correct": gold_cls == pred_cls,
                "claim_strength": f"{claim_strength_val:.6f}",
                "claim_base_score": metadata["claim_base_score"],
                "support_base_scores": metadata["support_base_scores"],
                "attack_base_scores": metadata["attack_base_scores"],
                "num_support": n_support,
                "num_attack": len(sample.attack),
                "qwen_argument_score_fallbacks": 0,
            }
        )
        all_preds.append(claim_strength_val)
        all_targets.append(sample.label)

    if csv_output_path is not None:
        _csv_path = Path(csv_output_path)
        _csv_path.parent.mkdir(parents=True, exist_ok=True)
        with _csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "row_idx",
                    "claim",
                    "gold_label",
                    "prediction",
                    "is_correct",
                    "claim_strength",
                    "claim_base_score",
                    "support_base_scores",
                    "attack_base_scores",
                    "num_support",
                    "num_attack",
                    "qwen_argument_score_fallbacks",
                ],
            )
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Wrote {len(csv_rows)} prediction rows to {_csv_path}")

    p = torch.tensor(all_preds)
    t = torch.tensor(all_targets)
    metrics: dict[str, float] = {
        "eval/mse": F.mse_loss(p, t).item(),
        "eval/mae": (p - t).abs().mean().item(),
    }

    pred_cls = _ternary_classify(all_preds, delta=delta)
    true_cls = _ternary_classify(all_targets, delta=delta)
    report: dict[str, Any] = classification_report(
        true_cls,
        pred_cls,
        labels=_CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )  # type: ignore[assignment]
    metrics["eval/accuracy"] = report["accuracy"]
    metrics["eval/macro_f1"] = report["macro avg"]["f1-score"]
    metrics["eval/macro_recall"] = report["macro avg"]["recall"]
    metrics["eval/macro_precision"] = report["macro avg"]["precision"]
    metrics["eval/balanced_accuracy"] = balanced_accuracy_score(true_cls, pred_cls)
    metrics["eval/hallucination_rate"] = hallucination_rate(true_cls, pred_cls)
    for cls_name in _CLASS_NAMES:
        metrics[f"eval/{cls_name}_precision"] = report[cls_name]["precision"]
        metrics[f"eval/{cls_name}_recall"] = report[cls_name]["recall"]
        metrics[f"eval/{cls_name}_f1"] = report[cls_name]["f1-score"]
    return metrics


def optimize_evaluate_with_scorer(
    samples: list[TrainingSample],
    semantics: GradualSemantics,
    score_fn: Scorer,
    estimate_claim_score: bool = False,
    csv_output_path: Path | str | None = None,
    criterion_id: str = "eval/balanced_accuracy",
) -> tuple[float, dict[str, float]]:
    """Optimizes the delta parameter and computes the associated metrics.

    Args:
        samples (list[TrainingSample]): Evaluation samples.
        semantics (GradualSemantics): Argumentation semantics to apply.
        score_fn (Scorer): Function mapping a list of conversations to a
            list of float scores in [0, 1].
        estimate_claim_score (bool): If True, also pass the claim through
            ``score_fn`` using the analyst topic prompt instead of fixing its
            base score to 0.5.
        csv_output_path (Path | str | None): If provided, write a per-sample
            prediction CSV to this path in the same format as tfu_predictions.csv.
        criterion_id (str): The metric to optimize for when selecting the best delta.

    Returns:
        (tuple[float, dict[str, float]]): The optimized delta value and the associated metrics.
    """
    scores, prediction_metadata = predict_with_scorer(
        samples, semantics, score_fn, estimate_claim_score=estimate_claim_score
    )

    best_delta = None
    best_metric_value = -float("inf")
    best_metrics: dict[str, float] = {}
    score_deltas = list(sorted([2 * abs(0.5 - s) for s in scores if s is not None]))
    delta_candidates = [
        (d1 + d2) / 2 for d1, d2 in zip(score_deltas, score_deltas[1:]) if d1 != d2
    ]
    for delta in delta_candidates:
        metrics = evaluate_with_scorer(
            samples,
            semantics,
            score_fn,
            estimate_claim_score=estimate_claim_score,
            scores=scores,
            prediction_metadata=prediction_metadata,
            delta=delta,
            csv_output_path=csv_output_path,
        )
        metric_value = metrics.get(criterion_id, -float("inf"))
        if metric_value > best_metric_value:
            best_metric_value = metric_value
            best_delta = delta
            best_metrics = metrics

    if best_delta is None:
        raise ValueError("No valid delta candidates found for optimization.")

    return best_delta, best_metrics


def evaluate_ranking(
    samples: list[RankingSample],
    score_fn: Scorer,
) -> dict[str, float]:
    """Compute mean Spearman rank correlation over all ranking samples.

    For each sample, arguments are scored (via ``score_fn`` if provided,
    otherwise via the regression head) and the resulting order is compared
    to the ground-truth ordering (index 0 = strongest argument).  Spearman's
    rho is +1 when the model perfectly reproduces the ground-truth ranking.

    Args:
        samples (list[RankingSample]): Ranking samples to evaluate on.
        score_fn (Scorer): Scoring function.

    Returns:
        (dict[str, float]): ``{"eval/mean_spearman": <float>}``.
    """
    spearman_scores: list[float] = []

    # Determine if we should use the base LLM prompt format
    use_base_llm_prompt = isinstance(score_fn, LLMScorer)

    for sample in samples:
        convs = [
            _build_conversation(
                sample.claim,
                arg,
                sample.is_support,
                use_base_llm_prompt=use_base_llm_prompt,
            )
            for arg in sample.arguments
        ]
        scores = score_fn(convs)
        # Compute Spearman rank correlation between predicted scores and ground-truth ranks
        # Lower list position = stronger argument, so gt ranks are in reverse order
        gt_ranks = list(range(len(sample.arguments), 0, -1))
        rho, _ = spearmanr(scores, gt_ranks)
        rho_float = float(rho)  # type: ignore
        if math.isnan(rho_float):
            rho_float = 0.0  # Treat NaN as 0 correlation
        spearman_scores.append(rho_float)

    mean_spearman = (
        sum(spearman_scores) / len(spearman_scores) if spearman_scores else 0.0
    )
    return {"eval/mean_spearman": mean_spearman}


def evaluate_consistency(
    samples: list[ConsistencySample],
    score_fn: Scorer,
) -> dict[str, float]:
    """Compute mean score variance over groups of semantically equivalent arguments.

    For each group of paraphrases, arguments are scored (via ``score_fn`` if
    provided, otherwise via the regression head) and the sample variance of
    those scores is recorded.  A perfectly consistent model assigns identical
    scores to all paraphrases, giving variance 0.

    Args:
        samples (list[ConsistencySample]): Consistency samples to evaluate on.
        score_fn (Scorer): Scoring function.

    Returns:
        (dict[str, float]): ``{"eval/mean_variance": <float>}``.
    """
    variances: list[float] = []

    # Determine if we should use the base LLM prompt format
    use_base_llm_prompt = isinstance(score_fn, LLMScorer)

    for sample in samples:
        convs = [
            _build_conversation(
                sample.claim,
                arg,
                sample.is_support,
                use_base_llm_prompt=use_base_llm_prompt,
            )
            for arg in sample.arguments
        ]
        scores = score_fn(convs)
        variances.append(float(np.var(scores, ddof=1)))

    mean_variance = sum(variances) / len(variances) if variances else 0.0
    return {"eval/mean_variance": mean_variance}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained LLMRegressionModel checkpoint on multiple validation datasets."
    )
    parser.add_argument(
        "--checkpoint-dir",
        help="Directory of the trained regression head checkpoint to load (required for --scoring-strategy regression).",
    )
    parser.add_argument(
        "--arg-data-path",
        required=False,
        help="Path to the JSON training data file.",
    )
    parser.add_argument(
        "--val-data-paths",
        nargs="+",
        required=True,
        metavar="PATH",
        help="One or more validation data files to evaluate on.",
    )
    parser.add_argument(
        "--ranking-data-paths",
        nargs="+",
        default=[],
        metavar="PATH",
        help="One or more ranking data files (JSONL) to evaluate rank correlation on.",
    )
    parser.add_argument(
        "--consistency-data-paths",
        nargs="+",
        default=[],
        metavar="PATH",
        help="One or more consistency data files (JSONL) to evaluate score variance on.",
    )
    parser.add_argument(
        "--semantics",
        default="dfq",
        choices=["dfq", "qe"],
        help="Gradual semantics to apply during evaluation.",
    )
    parser.add_argument(
        "--estimate-claim-scores",
        action="store_true",
        help=(
            "Estimate the claim's base score via the regression head using the "
            "analyst topic prompt (topic=True) instead of the hardcoded value "
            "of 0.5."
        ),
    )
    parser.add_argument(
        "--optimize-delta",
        action="store_true",
        help=(
            "If set, automatically optimize the delta parameter using the training data."
        ),
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=None,
        help=(
            "Parameter specifying the width of the 'uncertain' class. If None, "
            "the default boundaries of 1/3 and 2/3 are used. Ignored if "
            "--optimize-delta is set."
        ),
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
    parser.add_argument(
        "--scoring-strategy",
        default="regression",
        choices=["regression", "llm", "constant"],
        help=(
            "Strategy for scoring arguments: 'regression' (default) uses the "
            "trained regression head; 'llm' uses the backbone LLM's generated "
            "text (no regression head); 'constant' assigns every argument a "
            "fixed base score of 0.5."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for LLM generation (only used with --scoring-strategy llm).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="Model name to use when loading a base LLM (required with --scoring-strategy llm).",
    )
    parser.add_argument(
        "--csv-output-path",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to write a per-sample prediction CSV in the tfu_predictions.csv "
            "format. For a single val dataset this is the exact output path; for "
            "multiple val datasets per-dataset files are written as "
            "{parent}/{dataset_stem}_predictions.csv."
        ),
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log metrics to Weights & Biases.",
    )
    args = parser.parse_args()

    # Validate arguments
    if args.scoring_strategy == "llm" and not args.model_name:
        parser.error("--model-name is required when using --scoring-strategy llm")
    if args.scoring_strategy == "regression" and not args.checkpoint_dir:
        parser.error(
            "--checkpoint-dir is required when using --scoring-strategy regression"
        )
    if args.optimize_delta and not args.arg_data_path:
        parser.error("--arg-data-path is required when using --optimize-delta")

    random.seed(args.seed)

    progress_log = None
    if args.wandb:
        if args.checkpoint_dir:
            run_name = Path(args.checkpoint_dir).parent.name.replace(
                "trained_checkpoints_", ""
            )
        else:
            run_name = f"{args.scoring_strategy}_{args.semantics}"
        print(f"Initializing Weights & Biases run {run_name}...")
        wandb.init(
            project="argllms-val-regression-log-delta", name=run_name, config=vars(args)
        )
        progress_log = wandb.log

    # Load model and create scorer based on strategy
    score_fn: Scorer
    if args.scoring_strategy == "llm":
        print("Scoring strategy: base LLM generation (no regression head).")
        print(f"Loading model {args.model_name}...")
        model = LLMRegressionModel(
            model_name=args.model_name,
            lora_config=None,
            device_map=args.device_map,
            seed=args.seed,
        )
        score_fn = LLMScorer(model, batch_size=args.batch_size, seed=args.seed)
    elif args.scoring_strategy == "constant":
        print("Scoring strategy: constant base score of 0.5.")
        score_fn = ConstantScorer(score=0.5)
    else:  # regression
        print("Scoring strategy: regression head.")
        print(f"Loading model from {args.checkpoint_dir}...")
        model = LLMRegressionModel.load_checkpoint(
            args.checkpoint_dir,
            device_map=args.device_map,
            seed=args.seed,
        )
        score_fn = RegressionScorer(model)

    semantics = parse_semantics(args.semantics)

    # Optimize delta if requested
    if args.arg_data_path:
        print(
            f"\nEvaluating on training data from {args.arg_data_path} with delta={args.delta}..."
        )
        train_samples = load_training_samples(args.arg_data_path)
        metrics = evaluate_with_scorer(
            train_samples,
            semantics,
            score_fn,
            estimate_claim_score=args.estimate_claim_scores,
            delta=None,
            csv_output_path=None,
        )
        tag = Path(args.arg_data_path).stem
        print(
            f"Dataset [{tag}] (delta=None)  "
            f"MSE: {metrics['eval/mse']:.4f}  "
            f"MAE: {metrics['eval/mae']:.4f}  "
            f"Acc: {metrics['eval/accuracy']:.4f}  "
            f"Macro-F1: {metrics['eval/macro_f1']:.4f}  "
            f"Balanced Acc: {metrics['eval/balanced_accuracy']:.4f}"
        )

        if args.optimize_delta:
            print(
                f"\nOptimizing delta using training data from {args.arg_data_path}..."
            )
            best_delta, metrics = optimize_evaluate_with_scorer(
                train_samples,
                semantics,
                score_fn,
                estimate_claim_score=args.estimate_claim_scores,
                csv_output_path=None,
                criterion_id="eval/balanced_accuracy",
            )
            print(f"Best delta: {best_delta:.4f}")
            print(
                f"Dataset [{tag}] (delta={best_delta:.4f})  "
                f"MSE: {metrics['eval/mse']:.4f}  "
                f"MAE: {metrics['eval/mae']:.4f}  "
                f"Acc: {metrics['eval/accuracy']:.4f}  "
                f"Macro-F1: {metrics['eval/macro_f1']:.4f}  "
                f"Balanced Acc: {metrics['eval/balanced_accuracy']:.4f}"
            )

            if args.checkpoint_dir:
                delta_save_path = Path(args.checkpoint_dir).parent / "best_delta.json"
                delta_save_path.write_text(f'{{"best_delta": {best_delta:.6f}}}')
                print(f"Saved best delta to {delta_save_path}")

            args.delta = best_delta

    # Evaluate all validation datasets
    all_individual_metrics: list[dict[str, float]] = []

    for val_path in args.val_data_paths:
        val_samples = load_training_samples(val_path)
        print(f"\nEvaluating on {val_path} ({len(val_samples)} samples)...")

        csv_path: Path | None = None
        if args.csv_output_path is not None:
            if len(args.val_data_paths) == 1:
                csv_path = Path(args.csv_output_path)
            else:
                p = Path(args.csv_output_path)
                csv_path = p / f"{Path(val_path).stem}_predictions.csv"

        metrics = evaluate_with_scorer(
            val_samples,
            semantics,
            score_fn,
            args.estimate_claim_scores,
            csv_output_path=csv_path,
            delta=args.delta,
        )
        all_individual_metrics.append(metrics)

        # Print per-dataset results
        tag = Path(val_path).stem
        print(
            f"Dataset [{tag}]  "
            f"MSE: {metrics['eval/mse']:.4f}  "
            f"MAE: {metrics['eval/mae']:.4f}  "
            f"Acc: {metrics['eval/accuracy']:.4f}  "
            f"Macro-F1: {metrics['eval/macro_f1']:.4f}  "
            f"Balanced Acc: {metrics['eval/balanced_accuracy']:.4f}"
        )

        # Log per-dataset metrics to wandb
        if progress_log is not None:
            dataset_metrics = {f"val_{tag}/{k}": v for k, v in metrics.items()}
            progress_log(dataset_metrics)

    # Compute and print averaged metrics if multiple datasets
    if len(all_individual_metrics) > 1:
        print("\n--- Averaged metrics across all datasets ---")
        avg_metrics = average_metrics(all_individual_metrics)

        if avg_metrics:
            print(
                f"MSE: {avg_metrics.get('avg/mse', 0):.4f}  "
                f"MAE: {avg_metrics.get('avg/mae', 0):.4f}  "
                f"Acc: {avg_metrics.get('avg/accuracy', 0):.4f}  "
                f"Macro-F1: {avg_metrics.get('avg/macro_f1', 0):.4f}  "
                f"Balanced Acc: {avg_metrics.get('avg/balanced_accuracy', 0):.4f}"
            )

            if progress_log is not None:
                progress_log(avg_metrics)
    else:
        print("\nOnly one validation dataset; no averaging needed.")

    # Evaluate ranking datasets
    for ranking_path in args.ranking_data_paths:
        ranking_samples = load_ranking_samples(ranking_path)
        print(
            f"\nEvaluating ranking on {ranking_path} ({len(ranking_samples)} samples)..."
        )

        ranking_metrics = evaluate_ranking(ranking_samples, score_fn)
        tag = Path(ranking_path).stem
        print(
            f"Ranking [{tag}]  "
            f"Mean Spearman: {ranking_metrics['eval/mean_spearman']:.4f}"
        )

        if progress_log is not None:
            progress_log({f"ranking_{tag}/{k}": v for k, v in ranking_metrics.items()})

    # Evaluate consistency datasets
    for consistency_path in args.consistency_data_paths:
        consistency_samples = load_consistency_samples(consistency_path)
        print(
            f"\nEvaluating consistency on {consistency_path} ({len(consistency_samples)} samples)..."
        )

        consistency_metrics = evaluate_consistency(consistency_samples, score_fn)
        tag = Path(consistency_path).stem
        print(
            f"Consistency [{tag}]  "
            f"Mean Variance: {consistency_metrics['eval/mean_variance']:.6f}"
        )

        if progress_log is not None:
            progress_log(
                {f"consistency_{tag}/{k}": v for k, v in consistency_metrics.items()}
            )

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
