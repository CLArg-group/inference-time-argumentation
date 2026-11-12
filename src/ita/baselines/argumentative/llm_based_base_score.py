#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from ita.base_score_model.constants import CLAIM_IDX
from ita.baselines.direct.qwen3_base_prediction import (
    VALID_LABELS,
    apply_qwen_chat_template,
    extract_answer_text_from_raw_response,
    load_model,
    normalise_label,
)


DEFAULT_INPUT_PATHS = [
    "Data/Evaluation/evaluation_true_false_uncertain_with_arguments_qwen.json",
    "Data/Evaluation/evaluation_averitec_with_arguments_qwen.json",
]
LIKELIHOOD_PATTERN = re.compile(r"Likelihood:\s*(\d{1,3})\s*%?", re.IGNORECASE)


def require_analysis_dependencies():
    try:
        from ita.base_score_model.utils import parse_af_dict, parse_semantics
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing required analysis dependencies. Activate the project environment "
            "or run `uv sync --all-extras` before using this script."
        ) from exc

    return {
        "parse_af_dict": parse_af_dict,
        "parse_semantics": parse_semantics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyse generated argument frameworks with two baselines: "
            "Qwen per-argument base scores and constant base scores, both "
            "evaluated with the selected gradual semantics."
        )
    )
    parser.add_argument(
        "--input-paths",
        nargs="+",
        default=DEFAULT_INPUT_PATHS,
        help="One or more flat *_with_arguments.json files to analyse.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/baselines/qwen_argument_framework_score_analyses",
        help="Directory where scored JSON, evaluation CSVs, and summary JSON files are written.",
    )
    parser.add_argument(
        "--reuse-scored-dir",
        default=None,
        help=(
            "If set, skip Qwen rescoring and reuse existing "
            "'*_qwen_base_scores.json' and '*_constant_base_scores.json' files "
            "from this directory, rerunning only semantics/evaluation."
        ),
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3-8B",
        help="Hugging Face base model id or local LoRA adapter directory for Qwen base scoring.",
    )
    parser.add_argument(
        "--semantics",
        default="dfq",
        choices=["dfq", "qe"],
        help="Gradual semantics variant used to convert base scores into final claim strengths.",
    )
    parser.add_argument(
        "--upper-threshold",
        type=float,
        default=0.6666666,
        help="Threshold at or above which a claim is classified as True.",
    )
    parser.add_argument(
        "--lower-threshold",
        type=float,
        default=0.3333333,
        help="Threshold at or below which a claim is classified as False.",
    )
    parser.add_argument(
        "--constant-base-score",
        type=float,
        default=0.5,
        help="Constant base score used for the constant baseline.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Maximum total number of generated tokens per Qwen scoring prompt.",
    )
    parser.add_argument(
        "--generation-chunk-size",
        type=int,
        default=64,
        help="Number of tokens to request per Qwen generation step while waiting for a likelihood.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of framework rows processed from each input JSON.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N rows for each dataset. Use 0 to disable.",
    )
    parser.add_argument(
        "--save-raw-outputs",
        action="store_true",
        help="Store raw Qwen scoring outputs alongside the scored JSON.",
    )
    return parser.parse_args()


def build_argument_base_score_prompt(statement: str, claim: str, support: bool) -> str:
    return (
        "You are an analyst evaluating the validity and relevance of arguments. "
        f'For the argument:\n\nArgument: "{statement}"\n\nplease give your confidence '
        f"that the argument presents a compelling case {'in favour of' if support else 'against'} "
        f'the statement:\n\nStatement: "{claim}"\n\nYour assessment should be based '
        f"on how well the argument {'supports' if support else 'refutes'} the considered "
        "statement as well as the correctness, accuracy and truthfulness of the given argument. "
        "Your response should be between 0% and 100% with 0% indicating that the "
        "considered argument is definitely invalid, 100% indicating that the considered argument is "
        "definitely valid and values in between indicating various levels of "
        "uncertainty. Your estimates should be well-calibrated, so feel free to "
        "err on the side of caution and output moderate probabilities if you are "
        "not completely sure in your assessment. "
        "Please respond in the following form:"
        "\n\nLikelihood: The predicted likelihood that the considered argument is valid"
    )


def parse_likelihood(answer_text: str | None) -> float | None:
    if answer_text is None:
        return None

    stripped = answer_text.strip()
    if not stripped:
        return None

    match = LIKELIHOOD_PATTERN.search(stripped)
    if match is not None:
        value = int(match.group(1))
    else:
        first_line = stripped.splitlines()[0].strip()
        if first_line.isdigit():
            value = int(first_line)
        elif first_line.endswith("%") and first_line[:-1].strip().isdigit():
            value = int(first_line[:-1].strip())
        else:
            return None

    if not 0 <= value <= 100:
        return None
    return value / 100.0


def load_examples(input_path: Path, limit: int | None) -> list[dict[str, Any]]:
    with input_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a flat JSON list in {input_path}, found {type(data).__name__}."
        )

    examples = data[:limit] if limit is not None else data
    for index, example in enumerate(examples):
        if not isinstance(example, dict):
            raise ValueError(f"Example {index} in {input_path} is not a JSON object.")
        if "claim" not in example:
            raise ValueError(f"Example {index} in {input_path} is missing 'claim'.")
        if "label" not in example:
            raise ValueError(f"Example {index} in {input_path} is missing 'label'.")
        if "support" not in example or "attack" not in example:
            raise ValueError(
                f"Example {index} in {input_path} must contain both 'support' and 'attack'."
            )

    return examples


def load_scored_examples(input_path: Path, limit: int | None) -> list[dict[str, Any]]:
    with input_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a flat JSON list in {input_path}, found {type(data).__name__}."
        )

    examples = data[:limit] if limit is not None else data
    required_fields = {
        "claim",
        "label",
        "support",
        "attack",
        "claim_base_score",
        "support_base_scores",
        "attack_base_scores",
    }
    for index, example in enumerate(examples):
        if not isinstance(example, dict):
            raise ValueError(f"Example {index} in {input_path} is not a JSON object.")
        missing = sorted(field for field in required_fields if field not in example)
        if missing:
            raise ValueError(
                f"Example {index} in {input_path} is missing required fields: {', '.join(missing)}."
            )

    return examples


def decode_raw_response(tokenizer, output_ids: list[int]) -> str:
    return tokenizer.decode(output_ids, skip_special_tokens=False).strip()


def score_with_qwen(
    model,
    tokenizer,
    *,
    claim: str,
    argument: str,
    is_support: bool,
    max_new_tokens: int,
    generation_chunk_size: int,
) -> dict[str, Any]:
    import torch

    prompt = "/no_think" + build_argument_base_score_prompt(argument, claim, is_support) + "/no_think"
    text = apply_qwen_chat_template(
        tokenizer,
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
    )

    model_device = next(model.parameters()).device
    model_inputs = tokenizer(
        [text],
        return_tensors="pt",
        add_special_tokens=False,
    ).to(model_device)

    input_ids = model_inputs.input_ids
    attention_mask = model_inputs.attention_mask
    output_ids: list[int] = []
    eos_token_id = tokenizer.eos_token_id

    while len(output_ids) < max_new_tokens:
        remaining_budget = max_new_tokens - len(output_ids)
        step_tokens = min(generation_chunk_size, remaining_budget)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=step_tokens,
                do_sample=False,
                pad_token_id=eos_token_id,
            )

        new_token_ids = generated_ids[0][len(input_ids[0]):].tolist()
        if not new_token_ids:
            break

        output_ids.extend(new_token_ids)
        raw_response = decode_raw_response(tokenizer, output_ids)
        answer_text = extract_answer_text_from_raw_response(raw_response)
        likelihood = parse_likelihood(answer_text)
        if likelihood is not None:
            return {
                "score": likelihood,
                "parse_status": "parsed",
                "raw_output": raw_response,
                "answer_text": answer_text,
            }

        if eos_token_id is not None and eos_token_id in new_token_ids:
            break

        appended_ids = torch.tensor(
            [new_token_ids],
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
        appended_mask = torch.ones(
            (1, len(new_token_ids)),
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        input_ids = torch.cat([input_ids, appended_ids], dim=1)
        attention_mask = torch.cat([attention_mask, appended_mask], dim=1)

    raw_response = decode_raw_response(tokenizer, output_ids)
    answer_text = extract_answer_text_from_raw_response(raw_response)
    fallback_status = (
        "fallback_truncated_thinking"
        if "<think>" in raw_response and "</think>" not in raw_response
        else "fallback_to_0_5"
    )
    return {
        "score": 0.5,
        "parse_status": fallback_status,
        "raw_output": raw_response,
        "answer_text": answer_text,
    }


def strength_to_label(claim_strength: float, upper_threshold: float, lower_threshold: float) -> str:
    if claim_strength >= upper_threshold:
        return "True"
    if claim_strength <= lower_threshold:
        return "False"
    return "Uncertain"


def build_qwen_scored_example(
    model,
    tokenizer,
    example: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    claim = str(example["claim"])
    support_arguments = list(example.get("support", []))
    attack_arguments = list(example.get("attack", []))

    support_scores: list[float] = []
    support_statuses: list[str] = []
    support_raw_outputs: list[str] = []
    for argument in support_arguments:
        result = score_with_qwen(
            model=model,
            tokenizer=tokenizer,
            claim=claim,
            argument=str(argument),
            is_support=True,
            max_new_tokens=args.max_new_tokens,
            generation_chunk_size=args.generation_chunk_size,
        )
        support_scores.append(result["score"])
        support_statuses.append(result["parse_status"])
        if args.save_raw_outputs:
            support_raw_outputs.append(result["raw_output"])

    attack_scores: list[float] = []
    attack_statuses: list[str] = []
    attack_raw_outputs: list[str] = []
    for argument in attack_arguments:
        result = score_with_qwen(
            model=model,
            tokenizer=tokenizer,
            claim=claim,
            argument=str(argument),
            is_support=False,
            max_new_tokens=args.max_new_tokens,
            generation_chunk_size=args.generation_chunk_size,
        )
        attack_scores.append(result["score"])
        attack_statuses.append(result["parse_status"])
        if args.save_raw_outputs:
            attack_raw_outputs.append(result["raw_output"])

    scored_example = dict(example)
    scored_example["claim_base_score"] = 0.5
    scored_example["support_base_scores"] = support_scores
    scored_example["attack_base_scores"] = attack_scores
    scored_example["base_score_metadata"] = {
        "strategy": "qwen_llm_base_score",
        "model": args.model_id,
        "prompt_source": "src/ita/base_score_model/llm_basescore.py",
        "support_parse_statuses": support_statuses,
        "attack_parse_statuses": attack_statuses,
    }
    if args.save_raw_outputs:
        scored_example["base_score_metadata"]["support_raw_outputs"] = support_raw_outputs
        scored_example["base_score_metadata"]["attack_raw_outputs"] = attack_raw_outputs
    return scored_example


def build_constant_scored_example(
    example: dict[str, Any],
    constant_base_score: float,
) -> dict[str, Any]:
    scored_example = dict(example)
    scored_example["claim_base_score"] = constant_base_score
    scored_example["support_base_scores"] = [
        constant_base_score for _ in example.get("support", [])
    ]
    scored_example["attack_base_scores"] = [
        constant_base_score for _ in example.get("attack", [])
    ]
    scored_example["base_score_metadata"] = {
        "strategy": "constant_base_score",
        "base_score": constant_base_score,
    }
    return scored_example


def evaluate_scored_examples(
    scored_examples: list[dict[str, Any]],
    *,
    semantics_name: str,
    upper_threshold: float,
    lower_threshold: float,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    deps = require_analysis_dependencies()
    parse_af_dict = deps["parse_af_dict"]
    gradual_semantics = deps["parse_semantics"](semantics_name)

    evaluation_rows: list[dict[str, str]] = []
    correct = 0
    prediction_counts: Counter[str] = Counter()
    actual_counts: Counter[str] = Counter()
    confusion_counts: dict[str, Counter[str]] = {
        label: Counter() for label in VALID_LABELS
    }
    qwen_parse_fallback_arguments = 0

    for row_idx, example in enumerate(scored_examples):
        gold_label = normalise_label(example.get("label"))
        if gold_label not in VALID_LABELS:
            raise ValueError(f"Unexpected label value at row {row_idx}: {example.get('label')!r}")

        framework = {
            "support": list(example.get("support", [])),
            "attack": list(example.get("attack", [])),
        }
        arguments, _, adjacency_matrix = parse_af_dict(framework, str(example["claim"]))

        expected_support_len = len(framework["support"])
        expected_attack_len = len(framework["attack"])
        support_scores = list(example.get("support_base_scores", []))
        attack_scores = list(example.get("attack_base_scores", []))
        if len(support_scores) != expected_support_len or len(attack_scores) != expected_attack_len:
            raise ValueError(
                f"Base-score lengths do not match the framework at row {row_idx}."
            )

        base_scores = [float(example.get("claim_base_score", 0.5))]
        base_scores.extend(float(score) for score in support_scores)
        base_scores.extend(float(score) for score in attack_scores)
        if len(base_scores) != len(arguments):
            raise ValueError(
                f"Base-score vector length does not match parsed arguments at row {row_idx}."
            )

        claim_strength = float(
            gradual_semantics(
                adjacency_matrix.float(),
                json_to_tensor(base_scores),
            )[CLAIM_IDX]
        )
        predicted_label = strength_to_label(claim_strength, upper_threshold, lower_threshold)
        is_correct = predicted_label == gold_label
        correct += int(is_correct)

        prediction_counts[predicted_label] += 1
        actual_counts[gold_label] += 1
        confusion_counts[gold_label][predicted_label] += 1

        metadata = example.get("base_score_metadata", {})
        support_statuses = list(metadata.get("support_parse_statuses", []))
        attack_statuses = list(metadata.get("attack_parse_statuses", []))
        fallback_count = sum(
            1 for status in [*support_statuses, *attack_statuses] if status != "parsed"
        )
        qwen_parse_fallback_arguments += fallback_count

        evaluation_rows.append(
            {
                "row_idx": str(row_idx),
                "claim": str(example["claim"]),
                "gold_label": gold_label,
                "prediction": predicted_label,
                "is_correct": str(is_correct),
                "claim_strength": f"{claim_strength:.6f}",
                "claim_base_score": str(example.get("claim_base_score", 0.5)),
                "support_base_scores": json.dumps(support_scores),
                "attack_base_scores": json.dumps(attack_scores),
                "num_support": str(len(framework["support"])),
                "num_attack": str(len(framework["attack"])),
                "qwen_argument_score_fallbacks": str(fallback_count),
            }
        )

    total = len(scored_examples)
    accuracy = correct / total if total else 0.0
    summary = {
        "total_samples": total,
        "accuracy": accuracy,
        "correct": correct,
        "prediction_counts": dict(prediction_counts),
        "actual_counts": dict(actual_counts),
        "confusion_matrix": {
            gold: {pred: confusion_counts[gold].get(pred, 0) for pred in VALID_LABELS}
            for gold in VALID_LABELS
        },
        "qwen_argument_score_fallbacks": qwen_parse_fallback_arguments,
    }
    return evaluation_rows, summary


def json_to_tensor(values: list[float]):
    import torch

    return torch.tensor(values, dtype=torch.float32)


def write_json(output_path: Path, payload: Any) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def write_eval_csv(output_path: Path, rows: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
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
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def stem_without_with_arguments(input_path: Path) -> str:
    suffix = "_with_arguments"
    if input_path.stem.endswith(suffix):
        return input_path.stem[: -len(suffix)]
    return input_path.stem


def run_dataset(
    *,
    input_path: Path,
    output_dir: Path,
    model,
    tokenizer,
    args: argparse.Namespace,
) -> None:
    examples = load_examples(input_path, limit=args.limit)
    dataset_name = stem_without_with_arguments(input_path)

    print(f"\n[{input_path.name}] Loaded {len(examples)} framework examples")

    qwen_scored_examples: list[dict[str, Any]] = []
    for row_idx, example in enumerate(examples):
        qwen_scored_examples.append(
            build_qwen_scored_example(
                model=model,
                tokenizer=tokenizer,
                example=example,
                args=args,
            )
        )
        if args.progress_every and (row_idx + 1) % args.progress_every == 0:
            print(f"[{input_path.name}] Qwen base-score analysis {row_idx + 1}/{len(examples)}")

    constant_scored_examples = [
        build_constant_scored_example(example, args.constant_base_score)
        for example in examples
    ]

    qwen_eval_rows, qwen_summary = evaluate_scored_examples(
        qwen_scored_examples,
        semantics_name=args.semantics,
        upper_threshold=args.upper_threshold,
        lower_threshold=args.lower_threshold,
    )
    constant_eval_rows, constant_summary = evaluate_scored_examples(
        constant_scored_examples,
        semantics_name=args.semantics,
        upper_threshold=args.upper_threshold,
        lower_threshold=args.lower_threshold,
    )

    qwen_scored_path = output_dir / f"{dataset_name}_qwen_base_scores.json"
    constant_scored_path = output_dir / f"{dataset_name}_constant_base_scores.json"
    qwen_eval_path = output_dir / f"{dataset_name}_qwen_eval.csv"
    constant_eval_path = output_dir / f"{dataset_name}_constant_eval.csv"
    summary_path = output_dir / f"{dataset_name}_summary.json"

    write_json(qwen_scored_path, qwen_scored_examples)
    write_json(constant_scored_path, constant_scored_examples)
    write_eval_csv(qwen_eval_path, qwen_eval_rows)
    write_eval_csv(constant_eval_path, constant_eval_rows)
    write_json(
        summary_path,
        {
            "input_path": str(input_path),
            "semantics": args.semantics,
            "upper_threshold": args.upper_threshold,
            "lower_threshold": args.lower_threshold,
            "qwen_base_score": qwen_summary,
            "constant_base_score": constant_summary,
        },
    )

    print(f"[{input_path.name}] Qwen accuracy: {qwen_summary['accuracy']:.4f} ({qwen_summary['correct']}/{qwen_summary['total_samples']})")
    print(f"[{input_path.name}] Constant accuracy: {constant_summary['accuracy']:.4f} ({constant_summary['correct']}/{constant_summary['total_samples']})")
    print(f"[{input_path.name}] Wrote:")
    print(f"  {qwen_scored_path}")
    print(f"  {constant_scored_path}")
    print(f"  {qwen_eval_path}")
    print(f"  {constant_eval_path}")
    print(f"  {summary_path}")


def run_dataset_reusing_scores(
    *,
    input_path: Path,
    reuse_scored_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    dataset_name = stem_without_with_arguments(input_path)
    qwen_scored_path = reuse_scored_dir / f"{dataset_name}_qwen_base_scores.json"
    constant_scored_path = reuse_scored_dir / f"{dataset_name}_constant_base_scores.json"
    if not qwen_scored_path.exists():
        raise SystemExit(f"Saved Qwen scored JSON not found: {qwen_scored_path}")
    if not constant_scored_path.exists():
        raise SystemExit(f"Saved constant scored JSON not found: {constant_scored_path}")

    qwen_scored_examples = load_scored_examples(qwen_scored_path, args.limit)
    constant_scored_examples = load_scored_examples(constant_scored_path, args.limit)

    print(f"\n[{input_path.name}] Reusing saved scored JSON:")
    print(f"  {qwen_scored_path}")
    print(f"  {constant_scored_path}")

    qwen_eval_rows, qwen_summary = evaluate_scored_examples(
        qwen_scored_examples,
        semantics_name=args.semantics,
        upper_threshold=args.upper_threshold,
        lower_threshold=args.lower_threshold,
    )
    constant_eval_rows, constant_summary = evaluate_scored_examples(
        constant_scored_examples,
        semantics_name=args.semantics,
        upper_threshold=args.upper_threshold,
        lower_threshold=args.lower_threshold,
    )

    qwen_eval_path = output_dir / f"{dataset_name}_qwen_eval.csv"
    constant_eval_path = output_dir / f"{dataset_name}_constant_eval.csv"
    summary_path = output_dir / f"{dataset_name}_summary.json"

    write_eval_csv(qwen_eval_path, qwen_eval_rows)
    write_eval_csv(constant_eval_path, constant_eval_rows)
    write_json(
        summary_path,
        {
            "input_path": str(input_path),
            "reused_qwen_base_scores": str(qwen_scored_path),
            "reused_constant_base_scores": str(constant_scored_path),
            "semantics": args.semantics,
            "upper_threshold": args.upper_threshold,
            "lower_threshold": args.lower_threshold,
            "qwen_base_score": qwen_summary,
            "constant_base_score": constant_summary,
        },
    )

    print(
        f"[{input_path.name}] Qwen accuracy: {qwen_summary['accuracy']:.4f} "
        f"({qwen_summary['correct']}/{qwen_summary['total_samples']})"
    )
    print(
        f"[{input_path.name}] Constant accuracy: {constant_summary['accuracy']:.4f} "
        f"({constant_summary['correct']}/{constant_summary['total_samples']})"
    )
    print(f"[{input_path.name}] Wrote:")
    print(f"  {qwen_eval_path}")
    print(f"  {constant_eval_path}")
    print(f"  {summary_path}")


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_paths = [Path(input_path) for input_path in args.input_paths]
    reuse_scored_dir = Path(args.reuse_scored_dir) if args.reuse_scored_dir else None

    if reuse_scored_dir is None:
        for input_path in input_paths:
            if not input_path.exists():
                raise SystemExit(f"Input JSON not found: {input_path}")

        print(f"Loading Qwen scoring model: {args.model_id}")
        model, tokenizer = load_model(args.model_id)

        for input_path in input_paths:
            run_dataset(
                input_path=input_path,
                output_dir=output_dir,
                model=model,
                tokenizer=tokenizer,
                args=args,
            )
        return

    if not reuse_scored_dir.exists():
        raise SystemExit(f"Reuse scored directory not found: {reuse_scored_dir}")

    print(f"Reusing existing scored JSON from: {reuse_scored_dir}")
    for input_path in input_paths:
        run_dataset_reusing_scores(
            input_path=input_path,
            reuse_scored_dir=reuse_scored_dir,
            output_dir=output_dir,
            args=args,
        )


if __name__ == "__main__":
    main()
