"""Shared utilities for loading argument data in either the Data format or generated arguments format."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def load_raw_samples(data_path: str | Path) -> list[dict[str, Any]]:
    """Load samples from a data file, normalizing to a common dict format.

    Args:
        data_path (str | Path): Path to the JSON data file.

    Returns:
        (list[dict[str, Any]]): Normalized samples, each with keys
            ``"claim"`` (str), ``"support"`` (list[str]), ``"attack"``
            (list[str]), and ``"label"`` (str or ``None`` if absent).
    """
    data: Any = json.loads(Path(data_path).read_text())
    if isinstance(data, list):
        # New format: already a list of {claim, support, attack, label}
        for s in data:
            s["label"] = s["label"].lower()
        return data
    # Original format: top-level object with a "samples" list
    result: list[dict[str, Any]] = []
    for s in data.get("samples", []):
        if s.get("is_parse_error", False):
            continue
        result.append(
            {
                "claim": s["claim"],
                "support": s["arguments"].get("support", []),
                "attack": s["arguments"].get("attack", []),
                "label": s.get("label").lower(),
            }
        )
    return result


def load_argument_triples(
    data_path: str | Path,
) -> list[tuple[str, str, bool]]:
    """Parse a data file into ``(claim, argument, is_support)`` triples.

    Both supported file formats are accepted; auto-detection is applied.

    Args:
        data_path (str | Path): Path to the JSON data file.

    Returns:
        (list[tuple[str, str, bool]]): One triple per individual argument.
    """
    triples: list[tuple[str, str, bool]] = []
    for sample in load_raw_samples(data_path):
        claim: str = sample["claim"]
        for arg in sample.get("support", []):
            triples.append((claim, arg, True))
        for arg in sample.get("attack", []):
            triples.append((claim, arg, False))
    return triples


@dataclass
class RankingSample:
    """A single ranking sample with arguments ordered from best to worst."""

    claim: str
    arguments: list[str]
    is_support: bool


@dataclass
class ConsistencySample:
    """A group of semantically equivalent argument paraphrases for the same claim."""

    claim: str
    arguments: list[str]
    is_support: bool


def load_ranking_samples(data_path: str | Path) -> list[RankingSample]:
    """Load ranking samples from a JSONL file.

    Each line must be a JSON object with ``"statement"``, ``"is_effective_support"``
    (bool, not null), and ``"ground_truth_ranking"`` (list of at least two
    arguments ordered from best to worst).

    Args:
        data_path (str | Path): Path to the JSONL file.

    Returns:
        (list[RankingSample]): One sample per valid line.
    """
    samples: list[RankingSample] = []
    for line in Path(data_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        item: dict[str, Any] = json.loads(line)
        ranking: list[str] = item.get("ground_truth_ranking", [])
        if len(ranking) < 2:
            continue
        samples.append(
            RankingSample(
                claim=item["statement"],
                arguments=ranking,
                is_support=bool(item["is_effective_support"]),
            )
        )
    return samples


def load_consistency_samples(data_path: str | Path) -> list[ConsistencySample]:
    """Load consistency samples from a JSONL file.

    Each line must be a JSON object with ``"claim"``, ``"section"``
    (``"support"`` or ``"attack"``), and ``"expanded_arguments"`` (list of
    at least two semantically equivalent argument paraphrases).

    Args:
        data_path (str | Path): Path to the JSONL file.

    Returns:
        (list[ConsistencySample]): One sample per valid line.
    """
    samples: list[ConsistencySample] = []
    for line in Path(data_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        item: dict[str, Any] = json.loads(line)
        expanded: list[str] = item.get("expanded_arguments", [])
        if len(expanded) < 2:
            continue
        samples.append(
            ConsistencySample(
                claim=item["claim"],
                arguments=expanded,
                is_support=item["section"] == "support",
            )
        )
    return samples
