"""End-to-end training of the LLMRegressionModel regression head via argumentation semantics."""

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, balanced_accuracy_score
from torch.optim import AdamW
from tqdm import tqdm
import wandb

from ita.base_score_model.data_utils import (
    load_raw_samples,
    RankingSample,
    ConsistencySample,
    load_ranking_samples,
    load_consistency_samples,
)
from ita.base_score_model.llm_regression_model import (
    LoraAdapterConfig,
    LLMRegressionModel,
)
from ita.base_score_model.prime_regression_head import _build_conversation
from ita.base_score_model.utils import parse_semantics
from ita.semantics import GradualSemantics

# Index of the claim node in the argument framework (must be 0)
CLAIM_IDX = 0


@dataclass
class TrainingSample:
    claim: str
    support: list[str]
    attack: list[str]
    raw_label: str  # Original label string from the data file
    label: float  # 0.0 (False), 0.5 (Uncertain), or 1.0 (True)


# Normalise label strings regardless of case to a float in [0, 1].
_LABEL_MAP: dict[str, float] = {
    "true": 1.0,
    "false": 0.0,
    "uncertain": 0.5,
}


def load_training_samples(data_path: str | Path) -> list[TrainingSample]:
    """Load claim-with-arguments samples from a data file.

    Args:
        data_path (str | Path): Path to the JSON data file.

    Returns:
        (list[TrainingSample]): One sample per claim.
    """
    samples: list[TrainingSample] = []
    for item in load_raw_samples(data_path):
        raw_label = item.get("label")
        if raw_label is None:
            continue
        label = _LABEL_MAP.get(str(raw_label).lower())
        if label is None:
            continue
        samples.append(
            TrainingSample(
                claim=item["claim"],
                support=item["support"],
                attack=item["attack"],
                raw_label=raw_label,
                label=label,
            )
        )
    return samples


def _build_adjacency_matrix(n_support: int, n_attack: int) -> torch.Tensor:
    """Build an adjacency matrix for a claim with support and attack arguments.

    Layout: claim at index 0, support arguments at indices 1..n_support,
    attack arguments at indices n_support+1..n_support+n_attack.
    Positive entries indicate support; negative entries indicate attack.

    Args:
        n_support (int): Number of support arguments.
        n_attack (int): Number of attack arguments.

    Returns:
        (torch.Tensor): Float tensor of shape [1 + n_support + n_attack, 1 + n_support + n_attack].
    """
    n = 1 + n_support + n_attack
    A = torch.zeros(n, n)
    for i in range(1, 1 + n_support):
        A[i, CLAIM_IDX] = 1.0
    for i in range(1 + n_support, n):
        A[i, CLAIM_IDX] = -1.0
    return A


def _forward_sample(
    model: LLMRegressionModel,
    sample: TrainingSample,
    semantics: GradualSemantics,
    estimate_claim_score: bool = False,
) -> torch.Tensor:
    """Compute the predicted claim strength for a single sample.

    Builds conversations for all support and attack arguments, passes them
    through the regression head to obtain base scores, then applies the
    gradual semantics to obtain the final strength of the claim node.

    The claim's own base score is either fixed at 0.5 (default) or estimated
    by the regression head using the analyst topic prompt (when
    ``estimate_claim_score=True``).

    Args:
        model (LLMRegressionModel): Model whose regression head is being trained.
        sample (TrainingSample): The claim-argument sample.
        semantics (GradualSemantics): Differentiable argumentation semantics.
        estimate_claim_score (bool): If True, pass the claim through the
            regression head with the topic-level analyst prompt to obtain its
            base score rather than using the fixed value of 0.5.

    Returns:
        (torch.Tensor): Scalar tensor — the predicted claim strength in [0, 1].
    """
    support_convs = [
        _build_conversation(sample.claim, arg, True) for arg in sample.support
    ]
    attack_convs = [
        _build_conversation(sample.claim, arg, False) for arg in sample.attack
    ]
    arg_scores = model.tokenize_forward(
        support_convs + attack_convs
    )  # [n_support + n_attack]

    if estimate_claim_score:
        claim_conv = _build_conversation(sample.claim, topic=True)
        claim_score = model.tokenize_forward([claim_conv])  # [1]
    else:
        claim_score = torch.tensor(
            [0.5], dtype=torch.float32, device=model.input_device
        )
    base_scores = torch.cat([claim_score, arg_scores])  # [1 + n_support + n_attack]

    A = _build_adjacency_matrix(len(sample.support), len(sample.attack)).to(
        model.input_device
    )
    final_strengths = semantics.forward(A, base_scores, base_scores)
    return final_strengths[CLAIM_IDX]


def _ranking_loss(
    model: LLMRegressionModel,
    sample: RankingSample,
    margin: float,
) -> torch.Tensor:
    """Hinge pairwise ranking loss for a single ranking sample.

    For every pair (i, j) with i < j (argument i has a better ground-truth
    rank than argument j), penalises the model when the score margin falls
    below ``margin``:

        loss = mean over pairs of max(0, margin − (score_i − score_j))

    Args:
        model (LLMRegressionModel): Model whose regression head is scored.
        sample (RankingSample): Arguments ordered best (index 0) to worst.
        margin (float): Minimum required score gap between ranked pairs.

    Returns:
        (torch.Tensor): Scalar mean hinge loss over all ordered pairs.
    """
    convs = [
        _build_conversation(sample.claim, arg, sample.is_support)
        for arg in sample.arguments
    ]
    scores = model.tokenize_forward(convs)  # [n_args]
    pair_losses: list[torch.Tensor] = []
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            pair_losses.append(F.relu(margin - (scores[i] - scores[j])))
    return torch.stack(pair_losses).mean()


def _consistency_loss(
    model: LLMRegressionModel,
    sample: ConsistencySample,
) -> torch.Tensor:
    """Variance loss over semantically equivalent argument paraphrases.

    All arguments in ``sample`` should express the same meaning, so the
    regression head should assign them identical scores.  The loss is the
    sample variance of the predicted scores.

    Args:
        model (LLMRegressionModel): Model whose regression head is scored.
        sample (ConsistencySample): Semantically equivalent argument paraphrases.

    Returns:
        (torch.Tensor): Scalar variance of the predicted scores.
    """
    convs = [
        _build_conversation(sample.claim, arg, sample.is_support)
        for arg in sample.arguments
    ]
    scores = model.tokenize_forward(convs)  # [n_args]
    return scores.var()


def train_epoch(
    model: LLMRegressionModel,
    samples: list[TrainingSample],
    optimizer: AdamW,
    semantics: GradualSemantics,
    grad_accum_steps: int = 1,
    estimate_claim_score: bool = False,
    use_arg_loss: bool = True,
    ranking_samples: list[RankingSample] | None = None,
    ranking_margin: float = 0.1,
    consistency_samples: list[ConsistencySample] | None = None,
    progress_log: Callable | None = None,
):
    """Run one training epoch over all samples.

    When LoRA is active (``model._lora_used``), the backbone is set to train
    mode so that LoRA dropout is applied.  Otherwise it remains in eval mode
    to avoid unintended behaviour from any residual dropout layers.

    Gradients are accumulated over ``grad_accum_steps`` samples before each
    optimiser update.  The outer loop always iterates over the argumentative
    training ``samples``; for each iteration the enabled losses are computed
    and their gradients accumulated together:

    * **Argumentative loss** (``use_arg_loss=True``): MSE between the
      predicted claim strength and the ground-truth label.
    * **Ranking loss** (``ranking_samples`` provided): Hinge pairwise ranking
      loss for a randomly selected ranking sample.
    * **Consistency loss** (``consistency_samples`` provided): Variance of
      predicted scores for a randomly selected group of equivalent argument
      paraphrases.

    Args:
        model (LLMRegressionModel): Model to train.
        samples (list[TrainingSample]): Argumentative training samples that
            pace the iteration (one loop step per sample).
        optimizer (AdamW): Optimiser over trainable parameters.
        semantics (GradualSemantics): Differentiable argumentation semantics.
        grad_accum_steps (int): Number of samples to accumulate gradients
            over before an optimiser update.  Defaults to 1 (standard SGD).
        estimate_claim_score (bool): If True, estimate the claim's base score
            via the regression head (topic prompt) instead of fixing it at 0.5.
        use_arg_loss (bool): Whether to compute the argumentative MSE loss.
        ranking_samples (list[RankingSample] | None): If provided, a random
            sample is drawn each step and a hinge pairwise ranking loss is
            computed and back-propagated.
        ranking_margin (float): Margin for the hinge ranking loss.
        consistency_samples (list[ConsistencySample] | None): If provided, a
            random sample is drawn each step and a variance consistency loss
            is computed and back-propagated.
        progress_log (Callable | None): Optional step-level logging callback
            (e.g. ``wandb.log``).
    """
    # LoRA backbone needs train mode so its dropout is active; a fully frozen
    # backbone stays in eval mode.
    if model._lora_used:
        model.backbone.train()
    else:
        model.backbone.eval()
    model.regression_head.train()

    total_arg_loss = 0.0
    total_ranking_loss = 0.0
    total_consistency_loss = 0.0
    accum_arg_loss = 0.0
    accum_ranking_loss = 0.0
    accum_consistency_loss = 0.0
    n_samples = 0
    accum_count = 0
    optimizer.zero_grad()

    for sample in tqdm(samples, desc="Training"):
        had_loss = False

        if use_arg_loss and (sample.support or sample.attack):
            claim_strength = _forward_sample(
                model, sample, semantics, estimate_claim_score
            )
            label = torch.tensor(
                sample.label, dtype=torch.float32, device=model.input_device
            )
            loss = F.mse_loss(claim_strength, label) / grad_accum_steps
            loss.backward()
            accum_arg_loss += loss.item() * grad_accum_steps
            total_arg_loss += loss.item() * grad_accum_steps
            had_loss = True

        if ranking_samples is not None:
            rsample = random.choice(ranking_samples)
            rloss = _ranking_loss(model, rsample, ranking_margin) / grad_accum_steps
            rloss.backward()
            accum_ranking_loss += rloss.item() * grad_accum_steps
            total_ranking_loss += rloss.item() * grad_accum_steps
            had_loss = True

        if consistency_samples is not None:
            csample = random.choice(consistency_samples)
            closs = _consistency_loss(model, csample) / grad_accum_steps
            closs.backward()
            accum_consistency_loss += closs.item() * grad_accum_steps
            total_consistency_loss += closs.item() * grad_accum_steps
            had_loss = True

        if not had_loss:
            continue

        n_samples += 1
        accum_count += 1

        if accum_count == grad_accum_steps:
            optimizer.step()
            optimizer.zero_grad()

            log_dict: dict[str, float] = {}
            parts: list[str] = [f"Accumulation step {n_samples // grad_accum_steps}:"]
            if use_arg_loss:
                v = accum_arg_loss / grad_accum_steps
                parts.append(f"Arg: {v:.4f}")
                log_dict["train/step_arg_loss"] = v
            if ranking_samples is not None:
                v = accum_ranking_loss / grad_accum_steps
                parts.append(f"Ranking: {v:.4f}")
                log_dict["train/step_ranking_loss"] = v
            if consistency_samples is not None:
                v = accum_consistency_loss / grad_accum_steps
                parts.append(f"Consistency: {v:.4f}")
                log_dict["train/step_consistency_loss"] = v
            print("  ".join(parts))
            if progress_log is not None:
                progress_log(log_dict)

            accum_count = 0
            accum_arg_loss = 0.0
            accum_ranking_loss = 0.0
            accum_consistency_loss = 0.0

    # Flush any gradients accumulated from a partial final batch.
    if accum_count > 0:
        optimizer.step()
        optimizer.zero_grad()

        log_dict = {}
        parts = [f"Accumulation step {n_samples // grad_accum_steps + 1}:"]
        if use_arg_loss:
            v = accum_arg_loss / accum_count
            parts.append(f"Arg: {v:.4f}")
            log_dict["train/step_arg_loss"] = v
        if ranking_samples is not None:
            v = accum_ranking_loss / accum_count
            parts.append(f"Ranking: {v:.4f}")
            log_dict["train/step_ranking_loss"] = v
        if consistency_samples is not None:
            v = accum_consistency_loss / accum_count
            parts.append(f"Consistency: {v:.4f}")
            log_dict["train/step_consistency_loss"] = v
        print("  ".join(parts))
        if progress_log is not None:
            progress_log(log_dict)

    if progress_log is not None and n_samples > 0:
        epoch_log: dict[str, float] = {}
        if use_arg_loss:
            epoch_log["train/avg_arg_loss"] = total_arg_loss / n_samples
        if ranking_samples is not None:
            epoch_log["train/avg_ranking_loss"] = total_ranking_loss / n_samples
        if consistency_samples is not None:
            epoch_log["train/avg_consistency_loss"] = total_consistency_loss / n_samples
        if epoch_log:
            progress_log(epoch_log)


# Ternary class boundaries: [0, 1/3) → False, [1/3, 2/3) → Uncertain, [2/3, 1] → True.
_CLASS_NAMES = ["false", "uncertain", "true"]


def _ternary_classify(scores: list[float], delta: float | None = None) -> list[str]:
    """Map continuous scores in [0, 1] to ternary class labels.

    Boundaries at 1/3 and 2/3 give three equally-spaced classes.

    Args:
        scores (list[float]): Predicted or target scores.
        delta (float | None): parameter specifying the width of the "uncertain"
            class.  If None, the default boundaries of 1/3 and 2/3 are used.

    Returns:
        (list[str]): Class label per score: ``"false"``, ``"uncertain"``, or ``"true"``.
    """
    lower = 1 / 3 if delta is None else (0.5 - delta / 2)
    upper = 2 / 3 if delta is None else (0.5 + delta / 2)

    labels = []
    for s in scores:
        if s < lower:
            labels.append("false")
        elif s < upper:
            labels.append("uncertain")
        else:
            labels.append("true")
    return labels


def hallucination_rate(y_true: list[str], y_pred: list[str]) -> float:
    """Compute the hallucination rate, defined as the proportion of incorrect predictions that are not "uncertain".

    Args:
        y_true (list[str]): True class labels.
        y_pred (list[str]): Predicted class labels.

    Returns:
        float: Hallucination rate.
    """
    incorrect = 0
    total = 0
    for true, pred in zip(y_true, y_pred, strict=True):
        if true != pred and pred != "uncertain":
            incorrect += 1
        total += 1
    return incorrect / total if total > 0 else 0.0


def average_metrics(
    metrics_list: list[dict[str, float]], new_prefix: str = "avg/"
) -> dict[str, float]:
    """Compute averaged metrics across multiple evaluation runs.

    All metrics from evaluate() start with "eval/", and averaged versions
    will have the "eval/" prefix replaced with ``new_prefix``.

    Args:
        metrics_list: List of metric dictionaries from evaluate().
        new_prefix: Prefix to replace "eval/" with in the averaged metric keys
            (default: "avg/").

    Returns:
        Dictionary of averaged metrics with the specified prefix.
    """
    if not metrics_list:
        return {}

    averaged = {}
    # Collect all unique keys
    all_keys = set()
    for metrics in metrics_list:
        all_keys.update(metrics.keys())

    # Average each metric across datasets
    for key in sorted(all_keys):
        values = [m[key] for m in metrics_list if key in m]
        if values:
            # Replace "eval/" with the specified prefix in the key name
            avg_key = key.replace("eval/", new_prefix, 1)
            averaged[avg_key] = sum(values) / len(values)

    return averaged


def evaluate(
    model: LLMRegressionModel,
    samples: list[TrainingSample],
    semantics: GradualSemantics,
    estimate_claim_score: bool = False,
) -> dict[str, float]:
    """Compute regression and ternary classification metrics over the sample set.

    Classification uses equally-spaced thresholds (1/3, 2/3) to map
    continuous scores to False / Uncertain / True classes.  Per-class
    precision, recall and F1 are computed via ``sklearn``.

    Args:
        model (LLMRegressionModel): Model to evaluate.
        samples (list[TrainingSample]): Evaluation samples.
        semantics (GradualSemantics): Argumentation semantics.
        estimate_claim_score (bool): If True, estimate the claim's base score
            via the regression head (topic prompt) instead of fixing it at 0.5.

    Returns:
        (dict[str, float]): Regression metrics (``eval/mse``, ``eval/mae``)
            and per-class precision / recall / F1 plus overall accuracy and
            macro-F1.
    """
    model.eval()
    all_preds: list[float] = []
    all_targets: list[float] = []

    with torch.inference_mode():
        for sample in tqdm(samples, desc="Evaluating"):
            if not sample.support and not sample.attack:
                continue
            claim_strength = _forward_sample(
                model, sample, semantics, estimate_claim_score
            )
            all_preds.append(claim_strength.item())
            all_targets.append(sample.label)

    p = torch.tensor(all_preds)
    t = torch.tensor(all_targets)

    metrics: dict[str, float] = {
        "eval/mse": F.mse_loss(p, t).item(),
        "eval/mae": (p - t).abs().mean().item(),
    }

    # Ternary classification metrics via sklearn.
    pred_cls = _ternary_classify(all_preds)
    true_cls = _ternary_classify(all_targets)
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


def _run_evaluations(
    model: LLMRegressionModel,
    semantics: GradualSemantics,
    estimate_claim_score: bool,
    val_datasets: list[tuple[str, list[TrainingSample]]],
    test_datasets: list[tuple[str, list[TrainingSample]]],
    metrics: dict[str, Any],
) -> float | None:
    """Evaluate val and test splits, print a summary, and populate *metrics* in-place.

    Args:
        model (LLMRegressionModel): Model to evaluate.
        semantics (GradualSemantics): Argumentation semantics.
        estimate_claim_score (bool): Whether to estimate the claim's base score.
        val_datasets (list[tuple[str, list[TrainingSample]]]): (path, samples)
            pairs for each validation dataset.
        test_datasets (list[tuple[str, list[TrainingSample]]]): (path, samples)
            pairs for each test dataset.
        metrics (dict[str, Any]): Mutable dict updated in-place with prefixed
            metric keys (``"val_<stem>/...",`` ``"val_avg/...",`` and ``"test_<stem>/..."``).

    Returns:
        (float | None): Averaged val MSE if validation sets were provided, else None.
    """
    val_mse: float | None = None
    all_val_metrics: list[dict[str, float]] = []

    # Evaluate each validation dataset
    for val_path, val_samples in val_datasets:
        val_metrics = evaluate(model, val_samples, semantics, estimate_claim_score)
        all_val_metrics.append(val_metrics)
        tag = Path(val_path).stem
        metrics.update({f"val_{tag}/{k}": v for k, v in val_metrics.items()})
        print(
            f"Val [{tag}]  MSE: {val_metrics['eval/mse']:.4f}  "
            f"MAE: {val_metrics['eval/mae']:.4f}  "
            f"Acc: {val_metrics['eval/accuracy']:.4f}  "
            f"Macro-F1: {val_metrics['eval/macro_f1']:.4f}  "
            f"Balanced Acc: {val_metrics['eval/balanced_accuracy']:.4f}"
        )

    # Compute and log averaged validation metrics if multiple datasets
    if len(all_val_metrics) > 1:
        avg_metrics = average_metrics(all_val_metrics, new_prefix="val_avg/")
        metrics.update(avg_metrics)
        print(
            f"Val [Averaged]  MSE: {avg_metrics.get('val_avg/mse', 0):.4f}  "
            f"MAE: {avg_metrics.get('val_avg/mae', 0):.4f}  "
            f"Acc: {avg_metrics.get('val_avg/accuracy', 0):.4f}  "
            f"Macro-F1: {avg_metrics.get('val_avg/macro_f1', 0):.4f}  "
            f"Balanced Acc: {avg_metrics.get('val_avg/balanced_accuracy', 0):.4f}"
        )
        val_mse = avg_metrics.get("val_avg/mse")
    elif len(all_val_metrics) == 1:
        # For single validation set, use its MSE directly
        val_mse = all_val_metrics[0]["eval/mse"]

    # Evaluate test datasets
    for test_path, test_samples in test_datasets:
        test_metrics = evaluate(model, test_samples, semantics, estimate_claim_score)
        tag = Path(test_path).stem
        metrics.update({f"test_{tag}/{k}": v for k, v in test_metrics.items()})
        print(
            f"Test [{tag}]  MSE: {test_metrics['eval/mse']:.4f}  "
            f"MAE: {test_metrics['eval/mae']:.4f}  "
            f"Acc: {test_metrics['eval/accuracy']:.4f}  "
            f"Macro-F1: {test_metrics['eval/macro_f1']:.4f}  "
            f"Balanced Acc: {test_metrics['eval/balanced_accuracy']:.4f}"
        )
    return val_mse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the LLMRegressionModel regression head end-to-end on argumentation semantics."
    )
    parser.add_argument(
        "--arg-data-path",
        required=True,
        help="Path to the JSON training data file.",
    )
    parser.add_argument(
        "--val-data-paths",
        nargs="+",
        default=[],
        metavar="PATH",
        help="One or more validation data files to evaluate on.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Directory of the primed regression head checkpoint to load.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for saving trained checkpoints.",
    )
    parser.add_argument(
        "--semantics",
        default="dfq",
        choices=["dfq", "qe"],
        help="Gradual semantics to apply during training.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=8,
        help="Number of samples to accumulate gradients over before an optimiser update.",
    )
    parser.add_argument(
        "--lora",
        action="store_true",
        help="Apply LoRA to the backbone and fine-tune it alongside the regression head.",
    )
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank.")
    parser.add_argument(
        "--lora-alpha", type=int, default=32, help="LoRA scaling factor."
    )
    parser.add_argument(
        "--lora-dropout", type=float, default=0.1, help="LoRA dropout rate."
    )
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        nargs="+",
        default=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        help="Target modules for LoRA (e.g. 'q_proj k_proj v_proj o_proj').",
    )
    parser.add_argument(
        "--lora-bias",
        type=str,
        choices=["none", "all", "lora_only"],
        default="none",
        help="Whether and where to include bias parameters in LoRA (e.g. 'none', 'all', 'lora_only').",
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
        "--test-data-paths",
        nargs="+",
        default=[],
        metavar="PATH",
        help=(
            "One or more test data files to evaluate on (once per epoch and "
            "before training starts) without affecting training or checkpoint "
            "selection.  Both supported formats are accepted."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap the number of training samples (useful for debugging).",
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
        help="Random seed for better reproducibility.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log metrics to Weights & Biases.",
    )
    parser.add_argument(
        "--use-arg-loss",
        action="store_true",
        help="Whether to compute and back-propagate the argumentative MSE loss.",
    )
    parser.add_argument(
        "--ranking-data-path",
        default=None,
        help="Path to the ranking JSONL file (required when --use-ranking-loss is set).",
    )
    parser.add_argument(
        "--ranking-margin",
        type=float,
        default=0.1,
        help="Margin for the hinge pairwise ranking loss (default: 0.1).",
    )
    parser.add_argument(
        "--consistency-data-path",
        default=None,
        help="Path to the consistency JSONL file (required when --use-consistency-loss is set).",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # Validate loss selection.
    if (
        not args.use_arg_loss
        and args.ranking_data_path is None
        and args.consistency_data_path is None
    ):
        parser.error(
            "At least one loss and dataset must be specified via --use-arg-loss, "
            "--ranking-data-path, or --consistency-data-path."
        )

    progress_log = None
    if args.wandb:
        run_name = args.output_dir.split("/")[-1].replace("trained_checkpoints_", "")
        print(f"Initializing Weights & Biases run {run_name}...")
        wandb.init(
            project="argllms-train-regression-neo", config=vars(args), name=run_name
        )
        progress_log = wandb.log

    # Load data
    samples = load_training_samples(args.arg_data_path)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    print(f"Loaded {len(samples)} training samples.")

    val_datasets: list[tuple[str, list[TrainingSample]]] = []
    for vp in args.val_data_paths:
        vs = load_training_samples(vp)
        val_datasets.append((vp, vs))
        print(f"Loaded {len(vs)} validation samples from {vp}.")

    test_datasets: list[tuple[str, list[TrainingSample]]] = []
    for tp in args.test_data_paths:
        ts = load_training_samples(tp)
        test_datasets.append((tp, ts))
        print(f"Loaded {len(ts)} test samples from {tp}.")

    ranking_samples: list[RankingSample] | None = None
    if args.ranking_data_path is not None:
        ranking_samples = load_ranking_samples(args.ranking_data_path)
        print(
            f"Loaded {len(ranking_samples)} ranking samples from {args.ranking_data_path}."
        )

    consistency_samples: list[ConsistencySample] | None = None
    if args.consistency_data_path is not None:
        consistency_samples = load_consistency_samples(args.consistency_data_path)
        print(
            f"Loaded {len(consistency_samples)} consistency samples from {args.consistency_data_path}."
        )

    # Load model from primed checkpoint, optionally augmenting with LoRA.
    lora_adapter_cfg = (
        LoraAdapterConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
            bias=args.lora_bias,
        )
        if args.lora
        else None
    )
    model = LLMRegressionModel.load_checkpoint(
        args.checkpoint_dir,
        device_map=args.device_map,
        lora_config=lora_adapter_cfg,
        seed=args.seed,
    )

    if model._lora_used:
        trainable_params = list(model.regression_head.parameters()) + [
            p for p in model.backbone.parameters() if p.requires_grad
        ]
        print("LoRA enabled. Training LoRA adapters and regression head.")
    else:
        for param in model.backbone.parameters():
            param.requires_grad_(False)
        trainable_params = list(model.regression_head.parameters())
        print("Backbone frozen. Training regression head only.")

    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    semantics = parse_semantics(args.semantics)
    optimizer = AdamW(trainable_params, lr=args.lr)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_mse = float("inf")

    # Run evaluation before training to establish baseline performance
    print("\n--- Initial evaluation (before training) ---")
    initial_metrics: dict[str, Any] = {"epoch": 0}
    _run_evaluations(
        model,
        semantics,
        args.estimate_claim_scores,
        val_datasets,
        test_datasets,
        initial_metrics,
    )
    if progress_log is not None:
        progress_log(initial_metrics)

    # Train the model!
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1} / {args.epochs} ---")
        random.shuffle(samples)

        train_epoch(
            model,
            samples,
            optimizer,
            semantics,
            grad_accum_steps=args.grad_accum_steps,
            estimate_claim_score=args.estimate_claim_scores,
            use_arg_loss=args.use_arg_loss,
            ranking_samples=ranking_samples,
            ranking_margin=args.ranking_margin,
            consistency_samples=consistency_samples,
            progress_log=progress_log,
        )

        metrics: dict[str, Any] = {"epoch": epoch + 1}

        val_mse = _run_evaluations(
            model,
            semantics,
            args.estimate_claim_scores,
            val_datasets,
            test_datasets,
            metrics,
        )
        if val_mse is not None and val_mse < best_val_mse:
            # Save model checkpoint if validation MSE improved.
            best_val_mse = val_mse
            model.save_checkpoint(out_dir / "best_checkpoint")
            print(f"New best checkpoint saved (val MSE: {best_val_mse:.4f}).")

        if progress_log is not None:
            progress_log(metrics)

    # Always save the final state
    model.save_checkpoint(out_dir / "final_checkpoint")
    print("Final checkpoint saved.")

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
