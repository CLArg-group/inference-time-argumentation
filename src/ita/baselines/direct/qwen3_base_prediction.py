#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path


DEFAULT_INPUT_CSVS = [
    "Data/Evaluation/evaluation_true_false_uncertain.csv",
    "Data/Evaluation/evaluation_averitec.csv",
]
VALID_LABELS = ("True", "False", "Uncertain")
THINK_END_TOKEN_ID = 151668
LABEL_PATTERN = re.compile(r"\b(true|false|uncertain)\b", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a simple Qwen3-8B claim-classification baseline over one or more "
            "evaluation CSVs and save one prediction CSV per input file."
        )
    )
    parser.add_argument(
        "--input-csvs",
        nargs="+",
        default=DEFAULT_INPUT_CSVS,
        help=(
            "One or more CSV files containing a 'claim' column and either a 'valid' "
            "or 'label' gold-label column."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/baselines/qwen3_label_baseline",
        help="Directory where per-dataset prediction CSVs will be written.",
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3-8B",
        help="Hugging Face base model id or local LoRA adapter directory to load.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Maximum total number of generated tokens per claim.",
    )
    parser.add_argument(
        "--generation-chunk-size",
        type=int,
        default=64,
        help="Number of tokens to request per generation step while waiting for the final label.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of rows processed from each input CSV.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every N rows for each dataset. Use 0 to disable.",
    )
    return parser.parse_args()


def get_label_column(fieldnames: list[str] | None) -> str | None:
    if not fieldnames:
        return None

    for column in ("valid", "label"):
        if column in fieldnames:
            return column
    return None


def normalise_label(value: str | None) -> str | None:
    if value is None:
        return None

    match = LABEL_PATTERN.search(value.strip())
    if not match:
        return None

    return match.group(1).lower().capitalize()


def extract_answer_text_from_raw_response(raw_response: str | None) -> str | None:
    if raw_response is None:
        return None

    candidate = raw_response.strip()
    if not candidate:
        return None

    if "<think>" in candidate:
        if "</think>" not in candidate:
            return None
        candidate = candidate.rsplit("</think>", maxsplit=1)[-1].strip()

    if "<answer>" in candidate and "</answer>" in candidate:
        candidate = candidate.rsplit("<answer>", maxsplit=1)[-1]
        candidate = candidate.split("</answer>", maxsplit=1)[0].strip()

    return candidate or None


def extract_label_from_answer_text(value: str | None) -> str | None:
    if value is None:
        return None

    candidate = extract_answer_text_from_raw_response(value)
    if not candidate:
        return None

    return normalise_label(candidate)


def build_prompt(claim: str) -> str:
    return (
        "You are a careful fact-checking assistant.\n"
        "Classify the claim below using exactly one of the following three choice: "
        "True, False or Uncertain.\n"
        "Return only your choice a a single word and nothing else.\n\n"
        f"Claim: {claim}\n\n"
    )


def apply_qwen_chat_template(
    tokenizer,
    messages: list[dict[str, str]],
    add_generation_prompt: bool = True,
) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=True,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def _resolve_tokenizer_source(model_id: str, base_model_id: str) -> str:
    model_path = Path(model_id)
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]
    if model_path.is_dir() and any((model_path / name).exists() for name in tokenizer_files):
        return model_id
    return base_model_id


def load_model(model_id: str):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing required dependencies. Activate the project environment or run "
            "`uv sync --all-extras` before using this script."
        ) from exc

    model_path = Path(model_id)
    adapter_config_path = model_path / "adapter_config.json"

    if adapter_config_path.exists():
        try:
            from peft import PeftConfig, PeftModel
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "Loading a LoRA adapter requires `peft`. Install the training extras "
                "before evaluating adapter checkpoints."
            ) from exc

        peft_config = PeftConfig.from_pretrained(model_id)
        base_model_id = peft_config.base_model_name_or_path
        tokenizer_source = _resolve_tokenizer_source(model_id, base_model_id)

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, model_id)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


def get_think_end_token_ids(tokenizer) -> set[int]:
    token_ids = {THINK_END_TOKEN_ID}

    if hasattr(tokenizer, "convert_tokens_to_ids"):
        token_id = tokenizer.convert_tokens_to_ids("</think>")
        unk_token_id = getattr(tokenizer, "unk_token_id", None)
        if (
            isinstance(token_id, int)
            and token_id >= 0
            and token_id != unk_token_id
        ):
            token_ids.add(token_id)

    return token_ids


def find_last_think_end_position(tokenizer, output_ids: list[int]) -> int | None:
    think_end_ids = get_think_end_token_ids(tokenizer)
    last_position = None
    for index, token_id in enumerate(output_ids):
        if token_id in think_end_ids:
            last_position = index + 1
    return last_position


def decode_raw_response(tokenizer, output_ids: list[int]) -> str:
    return tokenizer.decode(output_ids, skip_special_tokens=False).strip()


def extract_answer_text(
    tokenizer,
    output_ids: list[int],
    raw_response: str | None = None,
) -> str | None:
    if raw_response:
        parsed_from_raw = extract_answer_text_from_raw_response(raw_response)
        if parsed_from_raw is not None:
            return parsed_from_raw
        if "<think>" in raw_response:
            return None

    boundary = find_last_think_end_position(tokenizer, output_ids)
    if boundary is not None:
        decoded = tokenizer.decode(
            output_ids[boundary:],
            skip_special_tokens=True,
        ).strip()
        return decoded or None

    decoded = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    if decoded.startswith("<think>"):
        return None
    return decoded or None


def generate_prediction(
    model,
    tokenizer,
    claim: str,
    max_new_tokens: int,
    generation_chunk_size: int = 64,
) -> dict[str, str]:
    import torch

    prompt = build_prompt(claim)
    messages = [{"role": "user", "content": prompt}]
    text = apply_qwen_chat_template(
        tokenizer,
        messages,
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
        answer_text = extract_answer_text(tokenizer, output_ids, raw_response)
        parsed_label = extract_label_from_answer_text(answer_text)

        if parsed_label is not None:
            break

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
    answer_text = extract_answer_text(tokenizer, output_ids, raw_response)

    return {
        "raw_response": raw_response,
        "answer_text": answer_text,
    }


def load_rows(input_path: Path, limit: int | None) -> tuple[list[dict[str, str]], str | None]:
    with input_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, str]] = []
        label_column = get_label_column(reader.fieldnames)

        for row_index, row in enumerate(reader):
            rows.append(row)
            if limit is not None and row_index + 1 >= limit:
                break

    return rows, label_column


def write_predictions(
    output_path: Path,
    input_fieldnames: list[str],
    predictions: list[dict[str, str]],
) -> None:
    fieldnames = [
        "row_idx",
        *input_fieldnames,
        "gold_label",
        "prediction",
        "parse_status",
        "is_correct",
        "answer_text",
        "raw_response",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)


def run_dataset(
    model,
    tokenizer,
    input_path: Path,
    output_dir: Path,
    max_new_tokens: int,
    generation_chunk_size: int,
    limit: int | None,
    progress_every: int,
) -> None:
    rows, label_column = load_rows(input_path, limit=limit)
    if not rows:
        print(f"[{input_path.name}] No rows found, skipping.")
        return

    if "claim" not in rows[0]:
        raise ValueError(f"{input_path} does not contain a 'claim' column.")

    output_path = output_dir / f"{input_path.stem}_predictions.csv"
    predictions: list[dict[str, str]] = []
    gold_available = label_column is not None
    correct = 0
    parse_fallbacks = 0
    prediction_counter: Counter[str] = Counter()

    for row_idx, row in enumerate(rows):
        response = generate_prediction(
            model=model,
            tokenizer=tokenizer,
            claim=row["claim"],
            max_new_tokens=max_new_tokens,
            generation_chunk_size=generation_chunk_size,
        )
        parsed_label = extract_label_from_answer_text(response["answer_text"])

        if parsed_label is None:
            parsed_label = "Uncertain"
            if "<think>" in response["raw_response"] and "</think>" not in response["raw_response"]:
                parse_status = "fallback_truncated_thinking"
            else:
                parse_status = "fallback_to_uncertain"
            parse_fallbacks += 1
        else:
            parse_status = "parsed"

        gold_label = normalise_label(row.get(label_column)) if gold_available else None
        is_correct = ""
        if gold_label is not None:
            is_correct = str(parsed_label == gold_label)
            correct += parsed_label == gold_label

        prediction_counter[parsed_label] += 1
        predictions.append(
            {
                "row_idx": str(row_idx),
                **row,
                "gold_label": gold_label or "",
                "prediction": parsed_label,
                "parse_status": parse_status,
                "is_correct": is_correct,
                "answer_text": response["answer_text"],
                "raw_response": response["raw_response"],
            }
        )

        if progress_every and (row_idx + 1) % progress_every == 0:
            print(f"[{input_path.name}] Processed {row_idx + 1}/{len(rows)} claims")

    input_fieldnames = list(rows[0].keys())
    write_predictions(output_path, input_fieldnames, predictions)

    print(f"\n[{input_path.name}] Saved predictions to {output_path}")
    print(f"[{input_path.name}] Total rows: {len(rows)}")
    print(f"[{input_path.name}] Prediction counts: {dict(prediction_counter)}")
    print(f"[{input_path.name}] Parse fallbacks: {parse_fallbacks}")
    if gold_available:
        accuracy = correct / len(rows)
        print(f"[{input_path.name}] Accuracy: {accuracy:.4f} ({correct}/{len(rows)})")


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_csv in args.input_csvs:
        if not Path(input_csv).exists():
            raise SystemExit(f"Input CSV not found: {input_csv}")

    print(f"Loading model: {args.model_id}")
    model, tokenizer = load_model(args.model_id)

    for input_csv in args.input_csvs:
        run_dataset(
            model=model,
            tokenizer=tokenizer,
            input_path=Path(input_csv),
            output_dir=output_dir,
            max_new_tokens=args.max_new_tokens,
            generation_chunk_size=args.generation_chunk_size,
            limit=args.limit,
            progress_every=args.progress_every,
        )


if __name__ == "__main__":
    main()
