import re
from dataclasses import dataclass
from typing import Optional

import torch

from ita.base_score_model.base_score_model import BaseScoreStrategy
from ita.base_score_model.constants import CLAIM_IDX
from ita.base_score_model.utils import parse_af_dict
from ita.semantics.gradual_semantics import GradualSemantics
from ita.grpo_finetuning.training_store import STORE, StepRecord


@dataclass
class RewardResult:
    reward: torch.Tensor
    predicted_class: Optional[str] = None
    actual_class: Optional[str] = None
    claim_strength: Optional[float] = None
    is_correct: Optional[bool] = None
    is_parse_error: bool = False
    num_supporters: int = 0
    num_attackers: int = 0
    thinking_tokens: int = 0
    mean_support_base_score: Optional[float] = None
    mean_attack_base_score: Optional[float] = None
    argument_scores: Optional[dict] = None  # {"support": [float, ...], "attack": [float, ...]}


def count_thinking_tokens(text: str) -> int:
    """Count words inside <think>…</think> blocks as a token-count proxy."""
    blocks = re.findall(r"<think>(.*?)</think>", text, re.DOTALL)
    return sum(len(b.split()) for b in blocks)


def accuracy_reward(
    claim: str,
    label: str,
    completion: dict,
    base_score_func: BaseScoreStrategy,
    gradual_semantics: GradualSemantics,
    upper_threshold: float,
    lower_threshold: float,
    raw_completion_text: str = "",   # original string before JSON parsing, for thinking-token count
    continuous_reward: bool = False,
) -> RewardResult:
    """
    Compute the argumentative reward for one completion.

    Args:
        claim:               The claim string being evaluated.
        label:               Ground-truth label ("TRUE" / "FALSE" / "UNCERTAIN").
        completion:          Already-parsed af_dict {"support": [...], "attack": [...]}.
        base_score_func:     A BaseScoreStrategy instance (constant / LLM / learned).
        gradual_semantics:   A GradualSemantics instance.
        upper_threshold:     Strength ≥ this → predicted TRUE.
        lower_threshold:     Strength ≤ this → predicted FALSE.
        raw_completion_text: Raw string from the model (used for thinking-token count).
    """
    step = STORE.get_step()
    thinking_tokens = count_thinking_tokens(raw_completion_text)


    try:
        af_dict = completion
        if ("support" not in af_dict) or ("attack" not in af_dict):
            raise ValueError("Missing 'support' or 'attack' keys")
        if not isinstance(af_dict["support"], list) or not isinstance(af_dict["attack"], list):
            raise ValueError("'support' and 'attack' must be lists")
    except Exception as e:
        print(f"JSON ERROR: {e}")
        failure_mode = f"json_parse_error: {type(e).__name__}"
        STORE.add_failure(failure_mode)
        STORE.add_record(StepRecord(
            step=step, claim=claim, label=label,
            predicted_strength=float("nan"),
            num_supporters=0, num_attackers=0,
            reward=-1000.0, thinking_tokens=thinking_tokens,
            failure_mode=failure_mode,
        ))
        return RewardResult(
            reward=torch.tensor(-1000.0),
            is_parse_error=True,
            thinking_tokens=thinking_tokens,
        )

    num_supporters = len(af_dict["support"])
    num_attackers = len(af_dict["attack"])

    # Flag empty generations
    if num_supporters == 0 and num_attackers == 0:
        failure_mode = "empty_generation"
        STORE.add_failure(failure_mode)
    else:
        failure_mode = None


    arguments, relations, adjacency_matrix = parse_af_dict(af_dict, claim)
    adjacency_matrix = adjacency_matrix.to(base_score_func.device)
    base_scores = base_score_func(arguments, relations)
    predicted_strength = gradual_semantics(adjacency_matrix, base_scores)[CLAIM_IDX]
    predicted_strength_val = float(predicted_strength)


    if predicted_strength_val >= upper_threshold:
        predicted = "TRUE"
    elif predicted_strength_val <= lower_threshold:
        predicted = "FALSE"
    else:
        predicted = "UNCERTAIN"

    is_correct = predicted.upper() == label.upper()

    if continuous_reward:
        # margin width = gap between the two thresholds (the "uncertain" band)
        if label.upper() == "UNCERTAIN":
            if lower_threshold <= predicted_strength_val <= upper_threshold:
                reward_value = 100.0
            elif predicted_strength_val < lower_threshold:
                reward_value = 100 - (200 / lower_threshold) * (lower_threshold - predicted_strength_val)
            else:
                reward_value = 100 - (200 / (1 - upper_threshold)) * (predicted_strength_val - upper_threshold)
        elif label.upper() == "TRUE":
            if predicted_strength_val >= upper_threshold:
                reward_value = 100.0
            else:
                reward_value = 200 * (predicted_strength_val / upper_threshold) - 100
        elif label.upper() == "FALSE":
            if predicted_strength_val <= lower_threshold:
                reward_value = 100.0
            else:
                reward_value = 200 * (1 - predicted_strength_val) / (1 - lower_threshold) - 100
        else:
            reward_value = 100.0 if is_correct else -100.0
    else:
        reward_value = 100.0 if is_correct else -100.0


    base_scores_list = base_scores.tolist()
    arg_log = []
    for idx, (arg_text, relation) in enumerate(zip(arguments, relations)):
        if idx == CLAIM_IDX:
            continue
        arg_log.append({
            "arg_id": f"{relation[0].upper()}{idx}",
            "type": relation,
            "text": arg_text,
            "base_score": base_scores_list[idx] if idx < len(base_scores_list) else float("nan"),
        })

    print("\n--- Argument Scores ---")
    print(f"Claim: {claim}")
    print(f"Ground Truth Label: {label}")
    for entry in arg_log:
        display = (entry["text"][:50] + "...") if len(entry["text"]) > 50 else entry["text"]
        print(f"  [{entry['type'].upper()}] Base Score={entry['base_score']:.4f} | {display}")
    print(f"Final Claim Strength: {predicted_strength_val:.4f}  →  Predicted: {predicted}")
    print("-----------------------")


    STORE.add_record(StepRecord(
        step=step,
        claim=claim,
        label=label,
        predicted_strength=predicted_strength_val,
        num_supporters=num_supporters,
        num_attackers=num_attackers,
        reward=reward_value,
        thinking_tokens=thinking_tokens,
        arguments=arg_log,
        failure_mode=failure_mode,
    ))

    support_scores = [e["base_score"] for e in arg_log if e["type"] == "support"]
    attack_scores  = [e["base_score"] for e in arg_log if e["type"] == "attack"]

    return RewardResult(
        reward=torch.tensor(reward_value),
        predicted_class=predicted.upper(),
        actual_class=label.upper(),
        claim_strength=predicted_strength_val,
        is_correct=is_correct,
        is_parse_error=False,
        num_supporters=num_supporters,
        num_attackers=num_attackers,
        thinking_tokens=thinking_tokens,
        mean_support_base_score=sum(support_scores) / len(support_scores) if support_scores else None,
        mean_attack_base_score=sum(attack_scores) / len(attack_scores) if attack_scores else None,
        argument_scores={"support": support_scores, "attack": attack_scores},
    )