from typing_extensions import override

import torch
from torch import Tensor

from ita.semantics import GradualSemantics


class DFQuadSemantics(GradualSemantics):
    def __init__(self, max_iters: int, conservativeness=1, epsilon: float = 0) -> None:
        super().__init__(max_iters, epsilon)
        self.conservativeness = conservativeness

    @override
    def aggregation_func(self, A: Tensor, strengths: Tensor) -> Tensor:

        # Product Aggregation

        supporter_mask = A == 1  # i supports j
        attacker_mask = A == -1  # i attacks j

        # (1 - state[i]) for each i
        inv_state = 1 - strengths  # shape (n,)

        # Expand to match adjacency matrix
        inv_state_matrix = inv_state.unsqueeze(1).expand_as(A)  # shape (n, n)

        # Compute products over rows i for each column j
        support_value = torch.prod(
            torch.where(attacker_mask, inv_state_matrix, torch.ones_like(A)), dim=0
        )
        attack_value = torch.prod(
            torch.where(supporter_mask, inv_state_matrix, torch.ones_like(A)), dim=0
        )
        result = support_value - attack_value

        return result

    @override
    def influence_func(self, base_scores, aggregations):
        """
        weight: tensor of shape (n,)   — previous strengths
        aggregate: tensor of shape (n,) — aggregated support-attack values
        conservativeness: scalar float

        Returns: tensor of shape (n,)
        """

        # Linear Influence

        # Boolean mask: aggregate > 0 elementwise
        pos_mask = aggregations > 0

        # Compute update when aggregate > 0
        positive_update = aggregations * (1 - base_scores) / self.conservativeness

        # Compute update when aggregate <= 0
        negative_update = aggregations * base_scores / self.conservativeness

        # Choose the correct update for each element
        update = torch.where(pos_mask, positive_update, negative_update)

        # Final strength
        result = base_scores + update

        return result
