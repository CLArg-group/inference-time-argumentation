import json
import re
from typing import Optional

import torch
import wandb
from datasets import load_dataset

from ita.base_score_model.base_score_model import BaseScoreStrategy
from ita.grpo_finetuning.reward_function import RewardResult, accuracy_reward
from ita.grpo_finetuning.threshold_optimization import ThresholdState
from ita.semantics.gradual_semantics import GradualSemantics
from ita.grpo_finetuning.training_store import STORE


# ---------------------------------------------------------------------------
# Auxiliary reward helpers (kept for reference / multi-reward experiments)
# ---------------------------------------------------------------------------

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion matches the expected format."""
    pattern = r"^<think>[\s\S]*?</think>\s*<answer>\s*\{[\s\S]*?\}\s*</answer>\s*$"
    completion_contents = [completion[0]["content"] for completion in completions]
    return [1.0 if re.match(pattern, c) else 0.0 for c in completion_contents]


def llm_output_verification(completions, **kwargs):
    """Reward function that checks if the model output contains valid JSON."""
    rewards = []
    for completion in completions:
        try:
            model_output = completion[0]["content"]
            json_match = re.search(
                r"</think>\s*<answer>\s*(\{[\s\S]*\})\s*</answer>\s*$",
                model_output,
                re.DOTALL,
            )
            if json_match:
                json.loads(json_match.group(1))
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        except Exception:
            rewards.append(0.0)
    return rewards


# ---------------------------------------------------------------------------
# Main reward closure
# ---------------------------------------------------------------------------

def argumentative_reward(
    base_score_func: BaseScoreStrategy,
    semantics: GradualSemantics,
    upper_threshold: float,
    lower_threshold: float,
    use_wandb: bool = False,
    continuous_reward: bool = False,
    threshold_state: Optional[ThresholdState] = None,
):
    """
    Returns a reward_function closure compatible with GRPOTrainer.

    Args:
        base_score_func:  A BaseScoreStrategy instance (constant / LLM / learned).
        semantics:        A GradualSemantics instance.
        upper_threshold:  Strength ≥ this → TRUE. Used as-is when threshold_state
                          is None; otherwise only used as the initial value.
        lower_threshold:  Strength ≤ this → FALSE. Same note as upper_threshold.
        use_wandb:        Whether to log per-batch metrics to W&B.
        threshold_state:  Optional live threshold holder (see
                          threshold_optimization.ThresholdState). When provided,
                          the closure reads current .upper/.lower on every call
                          instead of the static upper_threshold/lower_threshold
                          arguments, so a periodic callback can re-optimize
                          them mid-training. Defaults to None, which preserves
                          the original static-threshold behaviour exactly.
    """

    def reward_function(prompts, completions, **kwargs):
        rewards = []
        results = []

        current_upper = threshold_state.upper if threshold_state is not None else upper_threshold
        current_lower = threshold_state.lower if threshold_state is not None else lower_threshold

        # GRPOTrainer forwards extra dataset columns as lists in kwargs.
        # "claim" and "validity" were set as top-level keys in make_conversation.
        claims    = kwargs.get("claim",    [""] * len(completions))
        validities = kwargs.get("validity", [""] * len(completions))

        for i, completion in enumerate(completions):
            raw_text = completion[0]["content"]
            # Strip thinking block before JSON extraction
            completion_str = raw_text.split("</think>")[-1].strip()
            result = None

            try:
                import ast
                json_match = re.search(r"\{[\s\S]*\}", completion_str)
                if json_match is None:
                    raise ValueError("No JSON object found in completion after </think>")
                raw_json = json_match.group(0)
                try:
                    completion_data = json.loads(raw_json)
                except json.JSONDecodeError:
                    completion_data = ast.literal_eval(raw_json)

                # Unwrap nested "Output" key if present
                if "Output" in completion_data and isinstance(completion_data["Output"], dict):
                    completion_data = completion_data["Output"]

                print(f"\n--- Completion {i} ---")
                print(json.dumps(completion_data, indent=2))

                result = accuracy_reward(
                    claim=claims[i],
                    label=validities[i],
                    completion=completion_data,
                    base_score_func=base_score_func,
                    gradual_semantics=semantics,
                    upper_threshold=current_upper,
                    lower_threshold=current_lower,
                    raw_completion_text=raw_text,
                    continuous_reward=continuous_reward,
                )

                print(f"Reward: {result.reward.item():.2f}")
                print("-----------------------")

            except Exception as e:
                import traceback
                print(f"Error evaluating completion {i}: {e}|the completion was:\n{completion}")
                traceback.print_exc()
                print(raw_text)
                torch.cuda.empty_cache()
                result = RewardResult(reward=torch.tensor(-1000.0), is_parse_error=True)

            rewards.append(result.reward.unsqueeze(0))
            results.append(result)

        if not rewards:
            raise ValueError("No rewards were computed. Check input and reward calculation logic.")

        # ---------------------------------------------------------------- #
        # W&B batch-level metrics
        # ---------------------------------------------------------------- #
        if use_wandb:
            reward_values = [r.reward.item() for r in results]
            claim_strengths = [r.claim_strength for r in results if r.claim_strength is not None]
            parse_error_count = sum(1 for r in results if r.is_parse_error)
            valid_count = len(results) - parse_error_count
            correct_count = sum(1 for r in results if r.is_correct)

            support_bs = [r.mean_support_base_score for r in results if r.mean_support_base_score is not None]
            attack_bs  = [r.mean_attack_base_score  for r in results if r.mean_attack_base_score  is not None]

            metrics = {
                "reward/mean": sum(reward_values) / len(reward_values),
                "reward/min": min(reward_values),
                "reward/max": max(reward_values),
                "parse_errors": parse_error_count,
                "predictions/true": sum(1 for r in results if r.predicted_class == "TRUE"),
                "predictions/false": sum(1 for r in results if r.predicted_class == "FALSE"),
                "predictions/uncertain": sum(1 for r in results if r.predicted_class == "UNCERTAIN"),
                "thinking_tokens/mean": sum(r.thinking_tokens for r in results) / len(results),
                "arg_counts/supporters_mean": sum(r.num_supporters for r in results) / len(results),
                "arg_counts/attackers_mean": sum(r.num_attackers for r in results) / len(results),
                "thresholds/upper": current_upper,
                "thresholds/lower": current_lower,
            }
            if claim_strengths:
                metrics["claim_strength/mean"] = sum(claim_strengths) / len(claim_strengths)
            if support_bs:
                metrics["base_scores/supporters_mean"] = sum(support_bs) / len(support_bs)
            if attack_bs:
                metrics["base_scores/attackers_mean"] = sum(attack_bs) / len(attack_bs)
            if valid_count > 0:
                metrics["accuracy"] = correct_count / valid_count

            # Pin to the trainer's actual global_step (not wandb's own auto-increment
            # counter) — reward_function can be called more than once per optimizer
            # step (e.g. once per gradient-accumulation micro-batch), so without an
            # explicit step= these metrics drift out of sync with train/global_step
            # and any other step-pinned series (e.g. threshold_opt/*), producing a
            # visibly misaligned/shifted step function when plotted against it.
            wandb.log(metrics, step=STORE.get_step())

        return torch.cat(rewards, dim=0).cuda()

    return reward_function



def make_conversation(example):
    """Prepare conversation data for training."""
    SYSTEM_PROMPT = """
        Please provide a set of short arguments supporting and attacking the following claim.
        Construct the arguments so they refer to the truthfulness of the claim.
        The arguments should be short and concise, ideally one sentence per argument.
        The arguments should be based on factual information and logical reasoning.
        Only provide supporting/attacking arguments if you think there is a valid and convincing
        support/attack for this claim, otherwise return: N/A.
        Provide as many unique supporting/attacking arguments as you think are correct and applicable.

        Respond in the following format:

            claim:
                "<CLAIM>".
            Output:
                {
                    'support': [
                        "<SUPPORT ARGUMENT 1>",
                        "<SUPPORT ARGUMENT 2>",
                        ...,
                    ],
                    'attack': [
                        "<ATTACK ARGUMENT 1>",
                        "<ATTACK ARGUMENT 2>",
                        ...,
                    ]
                }
    """
    claim = example["claim"]
    user_content = f"Consider the following claim: {claim}\nand provide a response in the desired format."

    return {
        # Only well-formed chat messages go here — the tokenizer's chat template
        # will iterate over these and expects every entry to have "role" + "content".
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        # Metadata lives at the top level of the example, NOT inside prompt.
        # GRPOTrainer forwards all extra dataset columns as **kwargs to the
        # reward function, so we can read them there as kwargs["claim"] etc.
        "claim":    claim,
        "validity": example["valid"],
    }


def get_dataset(dataset_path: str):
    dataset = load_dataset("csv", data_files=dataset_path, split="train")
    train_dataset = dataset.map(make_conversation)
    return train_dataset, None