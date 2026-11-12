"""Online re-optimization of the GRPO reward's classification thresholds.

Reuses the same symmetric-delta parametrization already used to calibrate the
base-score model (see `ita.base_score_model.train_regression_head`'s
`_ternary_classify`: lower = 0.5 - delta/2, upper = 0.5 + delta/2), but
operates directly on the (claim_strength, gold_label) pairs already collected
during GRPO training in `training_store.STORE`, instead of recomputing scores
from raw text via a Scorer.

This is opt-in: nothing here is imported or executed unless
--enable-threshold-optimization is set.
"""
import json
import math
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from sklearn.metrics import accuracy_score, balanced_accuracy_score

from ita.base_score_model.train_regression_head import _ternary_classify
from ita.grpo_finetuning.training_store import StepRecord, ThresholdRecord

_CRITERIA = {
    "balanced_accuracy": balanced_accuracy_score,
    "accuracy": accuracy_score,
}


class ThresholdState:
    """Thread-safe mutable holder for the reward function's live thresholds.

    The reward closure reads `.upper`/`.lower` on every call; a periodic
    trainer callback calls `.update(...)` after re-optimizing.
    """

    def __init__(self, upper: float, lower: float):
        self._lock = threading.Lock()
        self._upper = upper
        self._lower = lower

    @property
    def upper(self) -> float:
        with self._lock:
            return self._upper

    @property
    def lower(self) -> float:
        with self._lock:
            return self._lower

    def update(self, upper: float, lower: float):
        with self._lock:
            self._upper = upper
            self._lower = lower


def optimize_thresholds_from_records(
    records: List[StepRecord],
    criterion: str = "balanced_accuracy",
    min_samples: int = 20,
) -> Optional[Tuple[float, float, float, Dict[str, float]]]:
    """Search for the best symmetric delta over a recent window of StepRecords.

    Mirrors `validate_regression_head.optimize_evaluate_with_scorer`'s search
    (exact candidates = midpoints between consecutive distinct observed
    |0.5 - score| * 2 values), but classifies directly against the gold
    TRUE/FALSE/UNCERTAIN label already stored per completion, since GRPO
    training already has the final claim-strength predictions on hand.

    Args:
        records: Candidate StepRecords (e.g. from STORE.get_records_since).
        criterion: Metric to maximize when selecting delta ("balanced_accuracy"
            or "accuracy").
        min_samples: Minimum number of valid (non-parse-error) records
            required to run the search.

    Returns:
        (new_upper, new_lower, best_delta, metrics) or None if there were too
        few valid samples / no valid delta candidates.
    """
    valid = [
        r
        for r in records
        if r.failure_mode is None
        and r.label.upper() in ("TRUE", "FALSE", "UNCERTAIN")
        and not math.isnan(r.predicted_strength)
    ]
    if len(valid) < min_samples:
        return None

    scores = [r.predicted_strength for r in valid]
    gold = [r.label.upper() for r in valid]

    score_deltas = sorted(2 * abs(0.5 - s) for s in scores)
    delta_candidates = [
        (d1 + d2) / 2 for d1, d2 in zip(score_deltas, score_deltas[1:]) if d1 != d2
    ]
    if not delta_candidates:
        return None

    scorer = _CRITERIA.get(criterion, balanced_accuracy_score)

    best_delta = None
    best_value = -float("inf")
    best_pred: Optional[List[str]] = None
    for delta in delta_candidates:
        pred = [c.upper() for c in _ternary_classify(scores, delta=delta)]
        value = scorer(gold, pred)
        if value > best_value:
            best_value = value
            best_delta = delta
            best_pred = pred

    if best_delta is None or best_pred is None:
        return None

    new_upper = 0.5 + best_delta / 2
    new_lower = 0.5 - best_delta / 2
    metrics = {
        "balanced_accuracy": balanced_accuracy_score(gold, best_pred),
        "accuracy": accuracy_score(gold, best_pred),
    }
    return new_upper, new_lower, best_delta, metrics


def load_threshold_history(threshold_log_path: Union[str, Path]) -> List[ThresholdRecord]:
    """Read a threshold_log.json (written by TrainingStore.save_threshold_log)
    and return its full history as ThresholdRecord objects, in original order.
    Returns an empty list if the file doesn't exist or is empty/unreadable.

    Used on resume: STORE is pure in-memory state and starts empty on every
    process restart. Without reloading the prior history:
      - the live thresholds would silently reset to the static config values
        instead of continuing from wherever they'd converged to before the
        crash/restart, and
      - the next save_threshold_log() call would overwrite the JSON log with
        only the new run's entries, silently dropping everything recorded
        before the restart.
    """
    path = Path(threshold_log_path)
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    return [ThresholdRecord(**entry) for entry in raw]
