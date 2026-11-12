from typing import List

import torch

from ita.base_score_model.constants import CLAIM_IDX
from ita.base_score_model.llm_basescore import base_score_func


class BaseScoreStrategy:
    def __init__(self, device: str):
        self.device = device

    def compute(self, arguments: List[str], relations: List[str]) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, arguments: List[str], relations: List[str]) -> torch.Tensor:
        return self.compute(arguments, relations)


class ConstantBaseScore(BaseScoreStrategy):
    def __init__(self, device: str, constant_base_score: float):
        super().__init__(device)
        self.constant_base_score = constant_base_score

    def compute(self, arguments: List[str], relations: List[str]) -> torch.Tensor:
        return (
            torch.ones((len(arguments)), device=self.device) * self.constant_base_score
        )


class LLMBaseScore(BaseScoreStrategy):
    def __init__(self, device):
        super().__init__(device)

    def compute(self, arguments: List[str], relations: List[str]) -> torch.Tensor:

        claim = arguments[CLAIM_IDX]
        base_scores = [0.5]
        for i in range(CLAIM_IDX + 1, len(arguments)):
            # TODO: Make the LLM Base Score more configurable (e.g. which LLM to use)
            base_scores.append(
                base_score_func(
                    claim, arguments[i], relations[i] == "support", device=self.device
                )
            )

        return torch.tensor(base_scores, device=self.device)


class LearnedBaseScoreModel(BaseScoreStrategy):
    """Base score strategy backed by LLMRegressionModel (safetensors checkpoint)."""

    def __init__(self, device: str, model):
        super().__init__(device)
        self.model = model

    def compute(self, arguments: List[str], relations: List[str]) -> torch.Tensor:
        from ita.base_score_model.prime_regression_head import _build_conversation

        claim = arguments[CLAIM_IDX]
        conversations = [
            _build_conversation(claim, arguments[i], relations[i] == "support")
            for i in range(CLAIM_IDX + 1, len(arguments))
        ]

        if not conversations:
            return torch.tensor([0.5], device=self.device)

        with torch.inference_mode():
            scores = self.model.tokenize_forward(conversations).tolist()

        base_scores = [0.5] + scores
        return torch.tensor(base_scores, device=self.device)


def build_learned_base_score_model(args):
    from ita.base_score_model.llm_regression_model import LLMRegressionModel
    model = LLMRegressionModel.load_checkpoint(
        args.bs_model_path,
        device_map={"": args.bs_device},
    )
    model.eval()
    return LearnedBaseScoreModel(args.bs_device, model)


def parse_base_score_model(args):
    match args.base_score_strategy:
        case "constant_base_score":
            base_score_func = ConstantBaseScore(
                device=args.bs_device, constant_base_score=args.base_score
            )
        case "llm_base_score":
            from ita.base_score_model import llm_basescore
            llm_basescore._bs_device = args.bs_device
            llm_basescore.load_model(device=args.bs_device)
            base_score_func = LLMBaseScore(
                device=args.bs_device,
            )
        case "learned_base_score":
            base_score_func = build_learned_base_score_model(args)
        case _:
            raise ValueError(
                f"No known base score model for {args.base_score_strategy}"
            )
    return base_score_func
