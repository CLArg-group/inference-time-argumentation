#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import random
from pathlib import Path

from ita.baselines.direct.qwen3_base_prediction import (
    DEFAULT_INPUT_CSVS,
    VALID_LABELS,
    build_prompt,
    extract_answer_text_from_raw_response,
    extract_label_from_answer_text,
    load_model,
    load_rows,
    normalise_label,
    run_dataset,
)


REWARD_CORRECT = 100.0
REWARD_INCORRECT = -100.0


def patch_vllm_guided_decoding_compatibility() -> None:
    try:
        import vllm.sampling_params as sampling_params
    except Exception:
        return

    if hasattr(sampling_params, "GuidedDecodingParams"):
        return

    structured_outputs_params = getattr(sampling_params, "StructuredOutputsParams", None)
    if structured_outputs_params is not None:
        sampling_params.GuidedDecodingParams = structured_outputs_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "LoRA-tune Qwen3-8B with GRPO for True/False/Uncertain classification "
            "and then evaluate the resulting adapter on the evaluation CSVs."
        )
    )
    parser.add_argument(
        "--train-csv",
        default="Data/Training/training_true_false_uncertain.csv",
        help="Training CSV containing 'claim' plus 'valid' or 'label'.",
    )
    parser.add_argument(
        "--eval-csvs",
        nargs="+",
        default=DEFAULT_INPUT_CSVS,
        help="One or more evaluation CSVs containing claims and gold labels.",
    )
    parser.add_argument(
        "--base-model-id",
        default="Qwen/Qwen3-8B",
        help="Base Hugging Face model id to tune with GRPO.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/baselines/qwen3_label_grpo",
        help="Directory where the LoRA adapter and evaluation outputs are written.",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=512,
        help="Maximum prompt length seen by GRPO.",
    )
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=512,
        help="Maximum sampled completion length during GRPO rollouts.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of GRPO training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate for LoRA GRPO training.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=1,
        help="Per-device batch size for GRPO training.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Gradient accumulation steps for GRPO training.",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=4,
        help="Number of sampled completions per prompt group for GRPO.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature used for GRPO rollouts.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="How often to log GRPO training progress.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for training.",
    )
    parser.add_argument(
        "--limit-train",
        type=int,
        default=None,
        help="Optional cap on the number of training rows to use.",
    )
    parser.add_argument(
        "--limit-eval",
        type=int,
        default=None,
        help="Optional cap on the number of rows processed per evaluation CSV.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Maximum total number of generated tokens per evaluation claim.",
    )
    parser.add_argument(
        "--generation-chunk-size",
        type=int,
        default=64,
        help="Number of tokens to request per generation step while waiting for the final label.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print evaluation progress every N rows for each dataset. Use 0 to disable.",
    )
    return parser.parse_args()


def require_training_dependencies():
    try:
        patch_vllm_guided_decoding_compatibility()

        from datasets import Dataset
        import torch
        from peft import LoraConfig, TaskType
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        from trl import GRPOConfig, GRPOTrainer
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing required GRPO dependencies. Activate the project environment "
            "or run `uv sync --all-extras` before using this script."
        ) from exc

    return {
        "Dataset": Dataset,
        "torch": torch,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "LoraConfig": LoraConfig,
        "TaskType": TaskType,
        "GRPOConfig": GRPOConfig,
        "GRPOTrainer": GRPOTrainer,
        "set_seed": set_seed,
    }


def choose_training_dtypes(torch_module) -> tuple[bool, bool, object]:
    if torch_module.cuda.is_available():
        if (
            hasattr(torch_module.cuda, "is_bf16_supported")
            and torch_module.cuda.is_bf16_supported()
        ):
            return True, False, torch_module.bfloat16
        return False, True, torch_module.float16
    return False, False, torch_module.float32


def build_grpo_chat_template_kwargs(tokenizer) -> dict[str, bool]:
    test_messages = [{"role": "user", "content": "test"}]
    try:
        tokenizer.apply_chat_template(
            test_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except TypeError:
        return {}
    return {"enable_thinking": True}


def build_training_dataset(dataset_cls, train_csv: Path, limit: int | None):
    rows, label_column = load_rows(train_csv, limit=limit)
    if not rows:
        raise ValueError(f"No rows found in {train_csv}.")
    if label_column is None:
        raise ValueError(f"{train_csv} does not contain a 'valid' or 'label' column.")

    examples: list[dict[str, object]] = []
    for row in rows:
        label = normalise_label(row.get(label_column))
        if label not in VALID_LABELS:
            raise ValueError(f"Unexpected label value: {row.get(label_column)!r}")

        examples.append(
            {
                "prompt": [{"role": "user", "content": build_prompt(row["claim"])}],
                "gold_label": label,
            }
        )

    return dataset_cls.from_list(examples)


def completion_to_text(completion) -> str:
    if isinstance(completion, str):
        return completion

    if isinstance(completion, list):
        assistant_chunks: list[str] = []
        for message in completion:
            if isinstance(message, dict) and message.get("role") == "assistant":
                assistant_chunks.append(str(message.get("content", "")))
        if assistant_chunks:
            return "\n".join(assistant_chunks)

    return str(completion)


def classification_reward(prompts, completions, gold_label, **kwargs) -> list[float]:
    rewards: list[float] = []

    for completion, expected_label in zip(completions, gold_label, strict=True):
        raw_completion = completion_to_text(completion)
        answer_text = extract_answer_text_from_raw_response(raw_completion)
        predicted_label = extract_label_from_answer_text(answer_text)
        expected = normalise_label(expected_label)

        if predicted_label is not None and predicted_label == expected:
            rewards.append(REWARD_CORRECT)
        else:
            rewards.append(REWARD_INCORRECT)

    return rewards


def main() -> None:
    args = parse_args()

    train_csv = Path(args.train_csv)
    if not train_csv.exists():
        raise SystemExit(f"Training CSV not found: {args.train_csv}")

    for eval_csv in args.eval_csvs:
        if not Path(eval_csv).exists():
            raise SystemExit(f"Evaluation CSV not found: {eval_csv}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_output_dir = output_dir / "eval_predictions"
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    deps = require_training_dependencies()
    Dataset = deps["Dataset"]
    torch = deps["torch"]
    AutoModelForCausalLM = deps["AutoModelForCausalLM"]
    AutoTokenizer = deps["AutoTokenizer"]
    LoraConfig = deps["LoraConfig"]
    TaskType = deps["TaskType"]
    GRPOConfig = deps["GRPOConfig"]
    GRPOTrainer = deps["GRPOTrainer"]
    set_seed = deps["set_seed"]

    random.seed(args.seed)
    set_seed(args.seed)

    print(f"Loading tokenizer: {args.base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_id,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Building GRPO dataset from {train_csv}")
    train_dataset = build_training_dataset(
        dataset_cls=Dataset,
        train_csv=train_csv,
        limit=args.limit_train,
    )
    print(f"Training samples: {len(train_dataset)}")

    use_bf16, use_fp16, torch_dtype = choose_training_dtypes(torch)
    chat_template_kwargs = build_grpo_chat_template_kwargs(tokenizer)

    print(f"Loading base model: {args.base_model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    training_args = GRPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_total_limit=1,
        remove_unused_columns=False,
        report_to="none",
        bf16=use_bf16,
        fp16=use_fp16,
        seed=args.seed,
        beta=0.0,
        chat_template_kwargs=chat_template_kwargs or None,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=classification_reward,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print("Starting GRPO training")
    trainer.train()

    print(f"Saving LoRA adapter and tokenizer to {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    del trainer
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Reloading GRPO adapter from {output_dir}")
    eval_model, eval_tokenizer = load_model(str(output_dir))

    for eval_csv in args.eval_csvs:
        run_dataset(
            model=eval_model,
            tokenizer=eval_tokenizer,
            input_path=Path(eval_csv),
            output_dir=eval_output_dir,
            max_new_tokens=args.max_new_tokens,
            generation_chunk_size=args.generation_chunk_size,
            limit=args.limit_eval,
            progress_every=args.progress_every,
        )


if __name__ == "__main__":
    main()
