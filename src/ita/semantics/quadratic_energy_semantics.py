from typing_extensions import override

import torch
from torch import Tensor

from ita.semantics import GradualSemantics


class QuadraticEnergySemantics(GradualSemantics):

    def __init__(self, max_iters: int, conservativeness=1, epsilon: float = 0) -> None:
        super().__init__(max_iters, epsilon)
        self.conservativeness = conservativeness

    @override
    def aggregation_func(self, A: Tensor, strengths: Tensor):
        return torch.matmul(torch.transpose(A, -2, -1), strengths)

    @override
    def influence_func(self, base_scores: Tensor, aggregations: Tensor):
        pos_mask = aggregations > 0

        scaled_aggregate = aggregations / self.conservativeness
        h = scaled_aggregate**2 / (1 + scaled_aggregate**2)

        positive_update = h * (1 - base_scores)
        negative_update = -h * base_scores

        update = torch.where(pos_mask, positive_update, negative_update)

        result = base_scores + update

        return result
