#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import random
from pathlib import Path

from ita.baselines.direct.qwen3_base_prediction import (
    DEFAULT_INPUT_CSVS,
    VALID_LABELS,
    apply_qwen_chat_template,
    build_prompt,
    get_think_end_token_ids,
    load_model,
    load_rows,
    normalise_label,
    run_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "LoRA-finetune Qwen3-8B to output True/False/Uncertain labels and "
            "then evaluate the resulting adapter on the evaluation CSVs."
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
        help="Base Hugging Face model id to fine-tune.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/baselines/qwen3_label_sft",
        help="Directory where the LoRA adapter and evaluation outputs are written.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum token length for training examples.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate for LoRA fine-tuning.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=1,
        help="Per-device batch size for training.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="How often to log training progress.",
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
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing required training dependencies. Activate the project environment "
            "or run `uv sync --all-extras` before using this script."
        ) from exc

    return {
        "torch": torch,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "LoraConfig": LoraConfig,
        "TaskType": TaskType,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
        "get_peft_model": get_peft_model,
        "set_seed": set_seed,
    }


def pad_to_max_length(
    input_ids: list[int],
    attention_mask: list[int],
    labels: list[int],
    pad_token_id: int,
    max_length: int,
) -> dict[str, list[int]]:
    pad_length = max_length - len(input_ids)
    if pad_length < 0:
        raise ValueError("Input is longer than max_length after truncation.")

    return {
        "input_ids": input_ids + [pad_token_id] * pad_length,
        "attention_mask": attention_mask + [0] * pad_length,
        "labels": labels + [-100] * pad_length,
    }


def mask_answer_only_labels(
    input_ids: list[int],
    prompt_length: int,
    think_end_token_ids: set[int],
) -> list[int]:
    labels = list(input_ids)
    for index in range(prompt_length):
        labels[index] = -100

    last_think_end = None
    for index in range(prompt_length, len(input_ids)):
        if input_ids[index] in think_end_token_ids:
            last_think_end = index

    if last_think_end is not None:
        for index in range(prompt_length, last_think_end + 1):
            labels[index] = -100

    if all(label == -100 for label in labels):
        raise ValueError(
            "No answer tokens remain after masking. Increase --max-length or check the chat template."
        )

    return labels


def encode_training_example(
    tokenizer,
    claim: str,
    label: str,
    max_length: int,
) -> dict[str, list[int]]:

    prompt_messages = [{"role": "user", "content": build_prompt(claim)}]
    full_messages = prompt_messages + [{"role": "assistant", "content": label}]

    prompt_text = apply_qwen_chat_template(
        tokenizer,
        prompt_messages,
        add_generation_prompt=True,
    )
    full_text = apply_qwen_chat_template(
        tokenizer,
        full_messages,
        add_generation_prompt=False,
    )

    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]
    full_ids = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]

    if len(full_ids) <= len(prompt_ids):
        raise ValueError(
            "The assistant answer was truncated away. Increase --max-length for training."
        )
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "Prompt tokenization is not a prefix of the full training example. "
            "Check the chat template for this tokenizer."
        )

    labels = mask_answer_only_labels(
        input_ids=full_ids,
        prompt_length=len(prompt_ids),
        think_end_token_ids=get_think_end_token_ids(tokenizer),
    )

    attention_mask = [1] * len(full_ids)
    return pad_to_max_length(
        input_ids=full_ids,
        attention_mask=attention_mask,
        labels=labels,
        pad_token_id=tokenizer.pad_token_id,
        max_length=max_length,
    )


class LabelSFTDataset:
    def __init__(self, encoded_examples: list[dict[str, list[int]]]):
        self.encoded_examples = encoded_examples

    def __len__(self) -> int:
        return len(self.encoded_examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.encoded_examples[index]


def build_training_dataset(
    tokenizer,
    train_csv: Path,
    max_length: int,
    limit: int | None,
) -> LabelSFTDataset:
    rows, label_column = load_rows(train_csv, limit=limit)
    if not rows:
        raise ValueError(f"No rows found in {train_csv}.")
    if label_column is None:
        raise ValueError(f"{train_csv} does not contain a 'valid' or 'label' column.")

    encoded_examples: list[dict[str, list[int]]] = []
    for row in rows:
        label = normalise_label(row.get(label_column))
        if label not in VALID_LABELS:
            raise ValueError(f"Unexpected label value: {row.get(label_column)!r}")

        encoded_examples.append(
            encode_training_example(
                tokenizer=tokenizer,
                claim=row["claim"],
                label=label,
                max_length=max_length,
            )
        )

    return LabelSFTDataset(encoded_examples)


def run_mask_sanity_check(dataset: LabelSFTDataset, tokenizer, limit: int = 3) -> None:
    for index in range(min(limit, len(dataset))):
        example = dataset[index]
        supervised_ids = [
            token_id
            for token_id, label in zip(example["input_ids"], example["labels"])
            if label != -100
        ]
        if not supervised_ids:
            raise ValueError(f"Training example {index} has no supervised tokens.")
        if any(token_id in get_think_end_token_ids(tokenizer) for token_id in supervised_ids):
            raise ValueError(
                f"Training example {index} still supervises </think> tokens."
            )

        supervised_text = tokenizer.decode(
            supervised_ids,
            skip_special_tokens=False,
        )
        if "Claim:" in supervised_text or "Classify the claim" in supervised_text:
            raise ValueError(
                f"Prompt tokens leaked into the supervised span for example {index}."
            )
        if not any(label in supervised_text for label in VALID_LABELS):
            raise ValueError(
                f"Supervised span for example {index} does not contain a gold label."
            )


def choose_training_dtypes(torch_module) -> tuple[bool, bool, object]:
    if torch_module.cuda.is_available():
        if (
            hasattr(torch_module.cuda, "is_bf16_supported")
            and torch_module.cuda.is_bf16_supported()
        ):
            return True, False, torch_module.bfloat16
        return False, True, torch_module.float16
    return False, False, torch_module.float32


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
    torch = deps["torch"]
    AutoModelForCausalLM = deps["AutoModelForCausalLM"]
    AutoTokenizer = deps["AutoTokenizer"]
    LoraConfig = deps["LoraConfig"]
    TaskType = deps["TaskType"]
    Trainer = deps["Trainer"]
    TrainingArguments = deps["TrainingArguments"]
    get_peft_model = deps["get_peft_model"]
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

    print(f"Building training dataset from {train_csv}")
    train_dataset = build_training_dataset(
        tokenizer=tokenizer,
        train_csv=train_csv,
        max_length=args.max_length,
        limit=args.limit_train,
    )
    print(f"Training samples: {len(train_dataset)}")
    run_mask_sanity_check(train_dataset, tokenizer)

    use_bf16, use_fp16, torch_dtype = choose_training_dtypes(torch)

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
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_total_limit=1,
        remove_unused_columns=False,
        report_to="none",
        bf16=use_bf16,
        fp16=use_fp16,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    print("Starting fine-tuning")
    trainer.train()

    print(f"Saving LoRA adapter and tokenizer to {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    del trainer
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Reloading fine-tuned adapter from {output_dir}")
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
