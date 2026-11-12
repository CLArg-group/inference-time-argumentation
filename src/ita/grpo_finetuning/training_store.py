import json
import threading
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class StepRecord:
    step: int
    claim: str
    label: str
    predicted_strength: float
    num_supporters: int
    num_attackers: int
    reward: float
    thinking_tokens: int
    arguments: List[Dict[str, Any]] = field(default_factory=list)
    failure_mode: Optional[str] = None


@dataclass
class ThresholdRecord:
    """One re-optimization event for the reward's upper/lower classification thresholds."""

    step: int
    upper: float
    lower: float
    delta: float
    metric_name: str
    metric_value: float
    n_samples: int


class TrainingStore:
    """Shared mutable state written by the reward fn, read by callbacks."""

    def __init__(self):
        self._lock = threading.Lock()
        self.records: List[StepRecord] = []
        self.loss_history: List[Dict] = []       # [{"step": int, "loss": float}]
        self.failure_counts: Dict[str, int] = defaultdict(int)
        self.threshold_history: List[ThresholdRecord] = []
        self._current_step: int = 0

    def set_step(self, step: int):
        with self._lock:
            self._current_step = step

    def get_step(self) -> int:
        with self._lock:
            return self._current_step

    def add_record(self, record: StepRecord):
        with self._lock:
            self.records.append(record)

    def add_loss(self, step: int, loss: float):
        with self._lock:
            self.loss_history.append({"step": step, "loss": loss})

    def add_failure(self, mode: str):
        with self._lock:
            self.failure_counts[mode] += 1

    def add_threshold_record(self, record: ThresholdRecord):
        with self._lock:
            self.threshold_history.append(record)

    def snapshot(self):
        """Return copies of all stored data (safe to read outside the lock)."""
        with self._lock:
            return (
                list(self.records),
                list(self.loss_history),
                dict(self.failure_counts),
            )

    def snapshot_thresholds(self) -> List[ThresholdRecord]:
        """Return a copy of the threshold re-optimization history."""
        with self._lock:
            return list(self.threshold_history)

    def get_records_since(self, min_step: int) -> List[StepRecord]:
        """Return copies of StepRecords with step >= min_step.

        Used to pull a recent window of completions (e.g. the last N training
        steps) for online threshold re-optimization.
        """
        with self._lock:
            return [r for r in self.records if r.step >= min_step]

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save_argument_log(self, log_path: Path):
        """Write every StepRecord as a JSONL line."""
        log_path.parent.mkdir(parents=True, exist_ok=True)
        records, _, _ = self.snapshot()
        with open(log_path, "w") as f:
            for r in records:
                f.write(json.dumps(asdict(r)) + "\n")

    def save_failure_summary(self, path: Path):
        """Dump failure_counts as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        _, _, failure_counts = self.snapshot()
        with open(path, "w") as f:
            json.dump(failure_counts, f, indent=2)

    def save_threshold_log(self, path: Path):
        """Dump the threshold re-optimization history as JSON.

        Writes an empty list when threshold optimization was never enabled,
        matching the no-op behaviour of the other save_* helpers.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        history = self.snapshot_thresholds()
        with open(path, "w") as f:
            json.dump([asdict(r) for r in history], f, indent=2)


STORE = TrainingStore()