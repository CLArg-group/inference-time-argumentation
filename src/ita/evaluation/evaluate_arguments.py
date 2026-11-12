import logging
from dataclasses import asdict

import torch
import numpy as np
import pandas as pd
import wandb
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from tqdm import tqdm

from ita.base_score_model.base_score_model import parse_base_score_model
from ita.base_score_model.utils import parse_semantics
from ita.evaluation.common import GenerationOutput
from ita.evaluation.parse_cli_evaluation import parse_evaluation_arguments
from ita.grpo_finetuning.reward_function import accuracy_reward

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_evaluation(args, use_wandb: bool = False):
    logger.info(f"Loading generated arguments from: {args.arguments_path}")
    generation_output = GenerationOutput.load(args.arguments_path)

    logger.info(f"Loaded {generation_output.metadata.total_samples} samples")
    logger.info(f"Parse errors from generation: {generation_output.metadata.parse_errors}")

    logger.info(f"Initializing base score model: {args.base_score_strategy}")
    base_score_func = parse_base_score_model(args)

    logger.info(f"Initializing gradual semantics: {args.semantics}")
    gradual_semantics = parse_semantics(args.semantics, args.conservativeness)

    predicted_labels = []
    actual_labels = []
    claim_strengths = []
    mean_support_scores = []
    mean_attack_scores = []
    skipped_parse_errors = 0
    correct_count = 0

    # Track parse error samples for inclusive metrics
    parse_error_actual_labels = []

    for sample in tqdm(generation_output.samples, desc="Evaluating"):
        if sample.is_parse_error:
            skipped_parse_errors += 1
            parse_error_actual_labels.append(sample.label)
            logger.warning(f"Skipping sample {sample.idx} due to parse error from generation")
            continue

        with torch.no_grad():
            result = accuracy_reward(
                claim=sample.claim,
                label=sample.label,
                completion=sample.arguments,
                base_score_func=base_score_func,
                gradual_semantics=gradual_semantics,
                upper_threshold=args.upper_threshold,
                lower_threshold=args.lower_threshold,
            )

        predicted_labels.append(result.predicted_class)
        actual_labels.append(result.actual_class)

        if result.claim_strength is not None:
            claim_strengths.append(result.claim_strength)

        mean_support_scores.append(result.mean_support_base_score)
        mean_attack_scores.append(result.mean_attack_base_score)
        sample.argument_scores = result.argument_scores
        generation_output.save(args.arguments_path)

        if result.is_correct:
            correct_count += 1
            print(f"Correct prediction for claim {sample.idx}")
        else:
            print(f"Incorrect prediction for claim {sample.idx}: Predicted={result.predicted_class}, Actual={result.actual_class}")

        if use_wandb:
            wandb.log({
                "sample_idx": sample.idx,
                "predicted_class": result.predicted_class,
                "actual_class": result.actual_class,
                "is_correct": 1 if result.is_correct else 0,
                "claim_strength": result.claim_strength,
            })

    # ===== METRICS EXCLUDING PARSE ERRORS =====
    total_evaluated = len(predicted_labels)
    accuracy_excl = correct_count / total_evaluated if total_evaluated > 0 else 0.0
    balanced_accuracy_excl = balanced_accuracy_score(actual_labels, predicted_labels) if predicted_labels else 0.0

    pred_true = sum(1 for p in predicted_labels if p == "TRUE")
    pred_false = sum(1 for p in predicted_labels if p == "FALSE")
    pred_uncertain = sum(1 for p in predicted_labels if p == "UNCERTAIN")

    actual_true = sum(1 for a in actual_labels if a == "TRUE")
    actual_false = sum(1 for a in actual_labels if a == "FALSE")
    actual_uncertain = sum(1 for a in actual_labels if a == "UNCERTAIN")

    strength_stats = {}
    if claim_strengths:
        strength_stats = {
            "claim_strength/mean": np.mean(claim_strengths),
            "claim_strength/min": np.min(claim_strengths),
            "claim_strength/max": np.max(claim_strengths),
            "claim_strength/std": np.std(claim_strengths),
        }

    cm_labels = ["FALSE", "UNCERTAIN", "TRUE"]
    cm_excl = None
    cm_excl_df = None
    if predicted_labels:
        cm_excl = confusion_matrix(actual_labels, predicted_labels, labels=cm_labels)
        cm_excl_df = pd.DataFrame(
            cm_excl,
            index=[f"true_{l}" for l in cm_labels],
            columns=[f"pred_{l}" for l in cm_labels],
        )

    # ===== METRICS INCLUDING PARSE ERRORS (as incorrect) =====
    # Parse errors are treated as incorrect predictions with "PARSE_ERROR" as predicted label
    all_predicted_labels = predicted_labels + ["PARSE_ERROR"] * skipped_parse_errors
    all_actual_labels = actual_labels + parse_error_actual_labels
    total_incl = len(all_predicted_labels)
    # correct_count stays the same since parse errors are incorrect
    accuracy_incl = correct_count / total_incl if total_incl > 0 else 0.0

    # Balanced accuracy inclusive: per-class recall across all samples (parse errors count as wrong)
    _incl_classes = ["TRUE", "FALSE", "UNCERTAIN"]
    _recalls_incl = []
    for _cls in _incl_classes:
        _total_cls = sum(1 for a in all_actual_labels if a == _cls)
        _correct_cls = sum(1 for a, p in zip(all_actual_labels, all_predicted_labels) if a == _cls and p == _cls)
        _recalls_incl.append(_correct_cls / _total_cls if _total_cls > 0 else 0.0)
    balanced_accuracy_incl = float(np.mean(_recalls_incl))

    # Confusion matrix including parse errors (add PARSE_ERROR as a prediction category)
    cm_labels_incl = ["FALSE", "UNCERTAIN", "TRUE", "PARSE_ERROR"]
    cm_incl = None
    cm_incl_df = None
    if all_predicted_labels:
        cm_incl = confusion_matrix(all_actual_labels, all_predicted_labels, labels=cm_labels_incl)
        cm_incl_df = pd.DataFrame(
            cm_incl,
            index=[f"true_{l}" for l in cm_labels_incl],
            columns=[f"pred_{l}" for l in cm_labels_incl],
        )

    # ===== OUTPUT RESULTS =====
    print(f"\n{'='*50}")
    print(f"METRICS EXCLUDING PARSE ERRORS")
    print(f"{'='*50}")
    if cm_excl_df is not None:
        print("\nConfusion Matrix (excluding parse errors):")
        print(cm_excl_df)
    print(f"\nTotal evaluated: {total_evaluated}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy_excl:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy_excl:.4f}")
    print(f"\nPredictions - TRUE: {pred_true}, FALSE: {pred_false}, UNCERTAIN: {pred_uncertain}")
    print(f"Actuals - TRUE: {actual_true}, FALSE: {actual_false}, UNCERTAIN: {actual_uncertain}")

    print(f"\n{'='*50}")
    print(f"METRICS INCLUDING PARSE ERRORS (as incorrect)")
    print(f"{'='*50}")
    if cm_incl_df is not None:
        print("\nConfusion Matrix (including parse errors):")
        print(cm_incl_df)
    print(f"\nTotal samples: {total_incl}")
    print(f"Parse errors: {skipped_parse_errors}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy_incl:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy_incl:.4f}")

    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    print(f"Total samples in file: {generation_output.metadata.total_samples}")
    print(f"Parse errors: {skipped_parse_errors}")
    print(f"Accuracy (excl. parse errors): {accuracy_excl:.4f}")
    print(f"Balanced Accuracy (excl. parse errors): {balanced_accuracy_excl:.4f}")
    print(f"Accuracy (incl. parse errors): {accuracy_incl:.4f}")
    print(f"Balanced Accuracy (incl. parse errors): {balanced_accuracy_incl:.4f}")
    if strength_stats:
        print(f"\nClaim Strength Stats:")
        for k, v in strength_stats.items():
            print(f"  {k}: {v:.4f}")

    if use_wandb:
        final_metrics = {
            # Metrics excluding parse errors
            "final/accuracy_excl_parse_errors": accuracy_excl,
            "final/balanced_accuracy_excl_parse_errors": balanced_accuracy_excl,
            "final/total_evaluated_excl": total_evaluated,
            # Metrics including parse errors
            "final/accuracy_incl_parse_errors": accuracy_incl,
            "final/balanced_accuracy_incl_parse_errors": balanced_accuracy_incl,
            "final/total_evaluated_incl": total_incl,
            # Common metrics
            "final/parse_errors": skipped_parse_errors,
            "final/correct_count": correct_count,
            "final/predictions_true": pred_true,
            "final/predictions_false": pred_false,
            "final/predictions_uncertain": pred_uncertain,
            "final/actuals_true": actual_true,
            "final/actuals_false": actual_false,
            "final/actuals_uncertain": actual_uncertain,
        }
        final_metrics.update({f"final/{k}": v for k, v in strength_stats.items()})
        wandb.log(final_metrics)

    # Save results CSV including parse error samples
    all_results = []
    for sample in generation_output.samples:
        if sample.is_parse_error:
            all_results.append({
                "idx": sample.idx,
                "claim": sample.claim,
                "actual_label": sample.label,
                "predicted_label": "PARSE_ERROR",
                "claim_strength": None,
                "mean_support_base_score": None,
                "mean_attack_base_score": None,
                "is_parse_error": True,
            })
        else:
            # Find the index in the evaluated samples
            eval_idx = len([s for s in generation_output.samples[:generation_output.samples.index(sample)] if not s.is_parse_error])
            all_results.append({
                "idx": sample.idx,
                "claim": sample.claim,
                "actual_label": actual_labels[eval_idx] if eval_idx < len(actual_labels) else None,
                "predicted_label": predicted_labels[eval_idx] if eval_idx < len(predicted_labels) else None,
                "claim_strength": claim_strengths[eval_idx] if eval_idx < len(claim_strengths) else None,
                "mean_support_base_score": mean_support_scores[eval_idx] if eval_idx < len(mean_support_scores) else None,
                "mean_attack_base_score": mean_attack_scores[eval_idx] if eval_idx < len(mean_attack_scores) else None,
                "is_parse_error": False,
            })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(args.output_path, index=False)
    logger.info(f"Saved evaluation results to: {args.output_path}")

    logger.info(f"Updated generated JSON with argument scores: {args.arguments_path}")

    return {
        "accuracy_excl_parse_errors": accuracy_excl,
        "balanced_accuracy_excl_parse_errors": balanced_accuracy_excl,
        "accuracy_incl_parse_errors": accuracy_incl,
        "balanced_accuracy_incl_parse_errors": balanced_accuracy_incl,
        "total_evaluated": total_evaluated,
        "total_samples": total_incl,
        "parse_errors": skipped_parse_errors,
    }


def main():
    print("Evaluating arguments...")

    args = parse_evaluation_arguments()

    use_wandb = not args.no_wandb
    if use_wandb:
        generation_output = GenerationOutput.load(args.arguments_path)

        wandb.init(
            project="argllms-eval",
            config={
                **vars(args),
                "generation_metadata": asdict(generation_output.metadata),
            },
        )

    run_evaluation(args, use_wandb)

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
