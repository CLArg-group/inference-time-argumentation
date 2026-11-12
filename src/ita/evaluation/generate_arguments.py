import json
import logging
import os
from datetime import datetime

import pandas as pd
import torch
import wandb
from peft import PeftConfig, PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from ita.evaluation.common import (
    GeneratedSample,
    GenerationMetadata,
    GenerationOutput,
)
from ita.evaluation.parse_cli_generation import parse_generation_arguments

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_model(args):
    model_id = args.model_id

    if os.path.exists(os.path.join(model_id, "adapter_config.json")):
        logger.info("LoRA adapter detected. Loading base model and adapter...")
        peft_config = PeftConfig.from_pretrained(model_id)
        base_model_path = peft_config.base_model_name_or_path

        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

        for parent in list(model.modules()):
            for child_name, child in list(parent.named_children()):
                if type(child).__name__ == "Gemma4ClippableLinear" and hasattr(child, "linear"):
                    setattr(parent, child_name, child.linear)

        model = PeftModel.from_pretrained(model, model_id)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

    model.to(args.device)

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


_GENERATION_CONFIG = {"max_new_tokens": 4096, "do_sample": False}


def _make_prompt(tokenizer, claim: str) -> str:
    from ita.evaluation.common import SYSTEM_PROMPT
    user_content = f"Consider the following claim: {claim}\nand provide response in a desired format."
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Fallback for tokenizers that don't support system role (e.g. Gemma base)
        messages = [{"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{user_content}"}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_batch(model, tokenizer, claims: list) -> list:
    tokenizer.padding_side = "left"
    prompts = [_make_prompt(tokenizer, c) for c in claims]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    input_length = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=GenerationConfig(**_GENERATION_CONFIG),
        )

    return [
        tokenizer.decode(output[input_length:], skip_special_tokens=True)
        for output in outputs
    ]


def generate_for_claim(model, tokenizer, claim: str) -> str:
    tokenizer.padding_side = "left"
    inputs = tokenizer(_make_prompt(tokenizer, claim), return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            generation_config=GenerationConfig(**_GENERATION_CONFIG),
        )

    return tokenizer.decode(output[0][input_length:], skip_special_tokens=True)


def parse_arguments_from_response(raw_response: str) -> dict:
    import ast
    import re as _re

    # Strip thinking block if present (Qwen3); no-op for models without it (Gemma)
    text = raw_response.split("</think>")[-1].strip()

    # Extract the first {...} block — handles any preamble before the JSON object
    json_match = _re.search(r"\{[\s\S]*\}", text)
    if json_match is None:
        raise ValueError("No JSON object found in response")
    raw_json = json_match.group(0)

    # json.loads requires double quotes; fall back to ast.literal_eval for single-quoted dicts
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(raw_json)

    # Unwrap nested "Output" key if present
    if "Output" in parsed and isinstance(parsed["Output"], dict):
        parsed = parsed["Output"]

    if not isinstance(parsed, dict):
        raise ValueError("Response is not a dictionary")
    if "support" not in parsed or "attack" not in parsed:
        raise ValueError("Response missing 'support' or 'attack' keys")
    if not isinstance(parsed["support"], list) or not isinstance(parsed["attack"], list):
        raise ValueError("'support' and 'attack' must be lists")

    return parsed


def get_label_column(df: pd.DataFrame, label_column: str) -> str:
    """Determine the label column name from the dataframe."""
    if label_column != "auto":
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found in CSV. Available columns: {list(df.columns)}")
        return label_column
    
    # Auto-detect label column
    for col in ["valid", "label"]:
        if col in df.columns:
            return col
    raise ValueError(f"Could not auto-detect label column. Available columns: {list(df.columns)}")


def _save_checkpoint(path: str, args, samples: list, parse_errors: int,
                     claims_with_retries: int, total_retries: int) -> None:
    total_samples = len(samples)
    parse_error_rate = parse_errors / total_samples if total_samples > 0 else 0.0
    avg_retries = total_retries / claims_with_retries if claims_with_retries > 0 else 0.0
    output = GenerationOutput(
        metadata=GenerationMetadata(
            model_id=args.model_id,
            eval_data_paths=args.eval_data_paths,
            timestamp=datetime.now().isoformat(),
            total_samples=total_samples,
            parse_errors=parse_errors,
            parse_error_rate=parse_error_rate,
            claims_with_retries=claims_with_retries,
            total_retries=total_retries,
            avg_retries=avg_retries,
        ),
        samples=samples,
    )
    output.save(path)


def run_generation(args, use_wandb: bool = False):
    logger.info(f"Loading model: {args.model_id}")
    model, tokenizer = load_model(args)

    # Load and concatenate all input CSV files
    all_data = []
    for eval_data_path in args.eval_data_paths:
        logger.info(f"Loading evaluation data: {eval_data_path}")
        df = pd.read_csv(eval_data_path)
        label_col = get_label_column(df, args.label_column)

        # Normalize label column name to 'label'
        if label_col != "label":
            df = df.rename(columns={label_col: "label"})

        df["source_file"] = eval_data_path
        all_data.append(df)

    eval_data = pd.concat(all_data, ignore_index=True)
    logger.info(f"Total claims to process: {len(eval_data)}")

    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created directory: {output_dir}")

    # Resume from existing output file if present.
    # Only treat successfully-parsed samples as done; parse-error entries are
    # retried so a fixed parser can recover them without full regeneration.
    if os.path.exists(args.output_path):
        existing = GenerationOutput.load(args.output_path)
        successful = [s for s in existing.samples if not s.is_parse_error]
        retrying   = [s for s in existing.samples if s.is_parse_error]
        samples = successful
        processed_indices = {s.idx for s in successful}
        parse_errors = 0          # reset — retried entries may now succeed
        claims_with_retries = existing.metadata.claims_with_retries
        total_retries = existing.metadata.total_retries
        eval_data = eval_data[~eval_data.index.isin(processed_indices)]
        logger.info(
            f"Resuming from {args.output_path}: "
            f"{len(successful)} already done, "
            f"{len(retrying)} parse-error entries will be retried, "
            f"{len(eval_data)} total remaining"
        )
    else:
        samples = []
        parse_errors = 0
        claims_with_retries = 0
        total_retries = 0

    max_attempts = 3
    batch_size = args.batch_size

    indices = eval_data.index.tolist()
    claims = eval_data["claim"].tolist()
    labels = eval_data["label"].tolist()
    n_batches = (len(eval_data) + batch_size - 1) // batch_size

    if n_batches == 0:
        logger.info("All claims already generated — nothing to do.")

    for batch_num in tqdm(range(n_batches), desc="Generating batches"):
        batch_slice = slice(batch_num * batch_size, (batch_num + 1) * batch_size)
        b_indices = indices[batch_slice]
        b_claims = claims[batch_slice]
        b_labels = labels[batch_slice]

        logger.info(f"Batch {batch_num + 1}/{n_batches}: generating {len(b_claims)} claims")
        raw_responses = generate_batch(model, tokenizer, b_claims)

        for idx, claim, label, raw_response in zip(b_indices, b_claims, b_labels, raw_responses):
            parsed_arguments = None
            last_error = None
            attempts = 1

            try:
                parsed_arguments = parse_arguments_from_response(raw_response)
            except Exception as e:
                last_error = e
                logger.warning(f"Parse error for claim {idx} (attempt 1/{max_attempts}): {e}. Retrying individually...")

                for retry in range(1, max_attempts):
                    attempts += 1
                    raw_response = generate_for_claim(model, tokenizer, claim)
                    try:
                        parsed_arguments = parse_arguments_from_response(raw_response)
                        break
                    except Exception as e2:
                        last_error = e2
                        if retry < max_attempts - 1:
                            logger.warning(f"Parse error for claim {idx} (attempt {attempts}/{max_attempts}): {e2}. Retrying...")
                        else:
                            logger.warning(f"Parse error for claim {idx} (attempt {attempts}/{max_attempts}): {e2}. No more retries.")

            if attempts > 1:
                claims_with_retries += 1
                total_retries += attempts - 1

            if parsed_arguments is not None:
                sample = GeneratedSample(
                    idx=idx,
                    claim=claim,
                    label=str(label).upper(),
                    arguments=parsed_arguments,
                    is_parse_error=False,
                )
                logger.info(f"Successfully generated arguments for claim {idx}" + (f" (after {attempts} attempts)" if attempts > 1 else ""))
            else:
                parse_errors += 1
                logger.warning(f"All {max_attempts} attempts failed for claim {idx}")
                print("############## PARSING ERROR ###############")
                print(f"Error details: {last_error}")
                print("Raw response:", raw_response)
                sample = GeneratedSample(
                    idx=idx,
                    claim=claim,
                    label=str(label).upper(),
                    arguments=None,
                    is_parse_error=True,
                    error_message=str(last_error),
                )

            samples.append(sample)
            _save_checkpoint(args.output_path, args, samples, parse_errors, claims_with_retries, total_retries)
            logger.info(f"Written {len(samples)} sample(s) to {args.output_path}")

            if use_wandb:
                wandb.log({
                    "sample_idx": idx,
                    "is_parse_error": 1 if sample.is_parse_error else 0,
                    "attempts": attempts,
                })

    total_samples = len(samples)
    parse_error_rate = parse_errors / total_samples if total_samples > 0 else 0.0
    avg_retries = total_retries / claims_with_retries if claims_with_retries > 0 else 0.0

    output = GenerationOutput(
        metadata=GenerationMetadata(
            model_id=args.model_id,
            eval_data_paths=args.eval_data_paths,
            timestamp=datetime.now().isoformat(),
            total_samples=total_samples,
            parse_errors=parse_errors,
            parse_error_rate=parse_error_rate,
            claims_with_retries=claims_with_retries,
            total_retries=total_retries,
            avg_retries=avg_retries,
        ),
        samples=samples,
    )

    output.save(args.output_path)
    logger.info(f"Saved generated arguments to: {args.output_path}")

    print(f"\n{'='*40}")
    print(f"Generation Complete")
    print(f"{'='*40}")
    print(f"Total samples: {total_samples}")
    print(f"Parse errors: {parse_errors}")
    print(f"Parse error rate: {parse_error_rate:.2%}")
    print(f"Claims needing retries: {claims_with_retries}")
    print(f"Total retries: {total_retries}")
    print(f"Average retries per claim (when retried): {avg_retries:.2f}")
    print(f"Output saved to: {args.output_path}")

    if use_wandb:
        wandb.log({
            "final/total_samples": total_samples,
            "final/parse_errors": parse_errors,
            "final/parse_error_rate": parse_error_rate,
            "final/claims_with_retries": claims_with_retries,
            "final/total_retries": total_retries,
            "final/avg_retries": avg_retries,
        })

    return output


def main():
    print("Generating arguments...")

    args = parse_generation_arguments()

    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project="argllms-generation",
            config=vars(args),
        )

    run_generation(args, use_wandb)

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
