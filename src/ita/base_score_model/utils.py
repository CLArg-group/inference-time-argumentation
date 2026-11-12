import torch

from ita.base_score_model.constants import CLAIM_IDX, EDGE_WEIGHTS
from ita.semantics import DFQuadSemantics, QuadraticEnergySemantics


def parse_af_dict(af_dict: dict, claim: str):
    no_args = 1 + len(af_dict["support"]) + len(af_dict["attack"])
    arguments = [claim]
    adjacency_matrix = torch.zeros((no_args, no_args))
    relations = ["claim"]  # Track relation type for each argument
    i = 1
    for supporter in af_dict["support"]:
        if isinstance(supporter, list):
            support_text = supporter[0]
        else:
            support_text = supporter

        arguments.append(support_text)
        relations.append("support")
        adjacency_matrix[i, CLAIM_IDX] = EDGE_WEIGHTS
        i += 1

    for attacker in af_dict["attack"]:
        if isinstance(attacker, list):
            attack_text = attacker[0]
        else:
            attack_text = attacker

        arguments.append(attack_text)
        relations.append("attack")
        adjacency_matrix[i, CLAIM_IDX] = -EDGE_WEIGHTS
        i += 1
    return arguments, relations, adjacency_matrix


def parse_semantics(semantics, conservativeness=1):

    match semantics:
        case "dfq":
            gradual_semantics = DFQuadSemantics(
                max_iters=3, epsilon=0, conservativeness=conservativeness
            )
        case "qe":
            gradual_semantics = QuadraticEnergySemantics(
                max_iters=3, epsilon=0, conservativeness=conservativeness
            )
        case _:
            raise ValueError(f"No known semantics for {semantics}")

    return gradual_semantics
