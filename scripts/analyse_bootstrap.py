"""BCa bootstrap confidence intervals for classification metrics from prediction CSV files.

Usage:

    python analyse_bootstrap.py path/to/tfu_predictions.csv [more CSVs ...]
    python analyse_bootstrap.py *.csv --confidence-level 0.99 --n-resamples 4999
"""

import argparse
import csv
from pathlib import Path
import warnings

import numpy as np
from scipy.stats import bootstrap
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

# Capitalised class names as written to CSV by validate_regression_head.py
_CLASS_NAMES = ["False", "Uncertain", "True"]
_DEFAULT_N_RESAMPLES = 10000
_DEFAULT_CONFIDENCE_LEVEL = 0.95


def load_predictions(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load gold labels and predictions from a prediction CSV file.

    Two CSV formats are supported and auto-detected from the header:

    * **Old format** (``validate_regression_head.py``): columns ``gold_label``
      and ``prediction``, labels already in title-case (``"False"``,
      ``"Uncertain"``, ``"True"``).
    * **New format** (``true_false_eval.csv``): columns ``actual_label``,
      ``predicted_label``, and ``is_parse_error``.  Labels are uppercase
      (``"FALSE"``, ``"UNCERTAIN"``, ``"TRUE"``) and are normalised to
      title-case on load.  Rows where ``is_parse_error`` is ``True`` are
      treated as always-incorrect regardless of the ``predicted_label`` value.

    Args:
        csv_path: Path to a prediction CSV file in either supported format.

    Returns:
        ``(y_true, y_pred)`` as numpy string arrays with title-case labels.
        Parse errors are represented by the sentinel ``"__PARSE_ERROR__"``,
        which does not match any valid class and is therefore always wrong.

    Raises:
        ValueError: If the CSV header does not match either known format.
    """
    gold_labels: list[str] = []
    predictions: list[str] = []
    with Path(csv_path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])

        if "gold_label" in fieldnames and "prediction" in fieldnames:
            # Old format: labels already title-cased, no parse-error column.
            for row in reader:
                gold_labels.append(row["gold_label"])
                predictions.append(row["prediction"])

        elif "actual_label" in fieldnames and "predicted_label" in fieldnames:
            # New format: uppercase labels + is_parse_error flag.
            for row in reader:
                gold_labels.append(row["actual_label"].capitalize())
                is_parse_error = row.get("is_parse_error", "").strip().lower() == "true"
                if is_parse_error:
                    # Sentinel guaranteed not to match any valid class label.
                    predictions.append("__PARSE_ERROR__")
                else:
                    predictions.append(row["predicted_label"].capitalize())

        else:
            raise ValueError(
                f"Unrecognised CSV format in {csv_path!r}. "
                "Expected columns ('gold_label', 'prediction') or "
                "('actual_label', 'predicted_label', 'is_parse_error')."
            )

    return np.array(gold_labels), np.array(predictions)


def _bca_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    statistic,
    n_resamples: int,
    confidence_level: float,
    random_state: int,
) -> tuple[float, float, float]:
    """Compute a point estimate and BCa bootstrap CI for a paired metric.

    Both arrays are resampled with the same indices on every bootstrap draw so
    that the (y_true, y_pred) pairing is preserved.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        statistic: Callable ``(y_true, y_pred) -> float``.
        n_resamples: Number of bootstrap resamples.
        confidence_level: Confidence level for the interval.
        random_state: Random seed.

    Returns:
        ``(point_estimate, ci_low, ci_high)``.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="y_pred contains classes not in y_true",
            category=UserWarning,
        )
        point = float(statistic(y_true, y_pred))
        rng = np.random.default_rng(random_state)
        result = bootstrap(
            (y_true, y_pred),
            statistic=statistic,
            paired=True,
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            method="BCa",
            rng=rng,
        )
    return (
        point,
        float(result.confidence_interval.low),
        float(result.confidence_interval.high),
    )


def _build_metric_list() -> list[tuple[str, object]]:
    """Return an ordered list of ``(name, statistic)`` pairs to bootstrap."""
    metrics: list[tuple[str, object]] = [
        ("accuracy", accuracy_score),
        (
            "balanced_accuracy",
            balanced_accuracy_score,
        ),
        (
            "macro_f1",
            lambda yt, yp: f1_score(
                yt, yp, labels=_CLASS_NAMES, average="macro", zero_division=0
            ),
        ),
        (
            "macro_precision",
            lambda yt, yp: precision_score(
                yt, yp, labels=_CLASS_NAMES, average="macro", zero_division=0
            ),
        ),
        (
            "macro_recall",
            lambda yt, yp: recall_score(
                yt, yp, labels=_CLASS_NAMES, average="macro", zero_division=0
            ),
        ),
    ]
    return metrics


def compute_bootstrap_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_resamples: int = _DEFAULT_N_RESAMPLES,
    confidence_level: float = _DEFAULT_CONFIDENCE_LEVEL,
    random_state: int = 42,
) -> dict[str, tuple[float, float, float]]:
    """Compute all metrics with BCa bootstrap CIs.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        n_resamples: Number of bootstrap resamples.
        confidence_level: Confidence level for the intervals.
        random_state: Random seed.

    Returns:
        Dict mapping metric name to ``(point, ci_low, ci_high)``.
    """
    results: dict[str, tuple[float, float, float]] = {}
    for name, fn in tqdm(_build_metric_list(), desc="Bootstrap CI"):
        results[name] = _bca_ci(
            y_true,
            y_pred,
            fn,
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            random_state=random_state,
        )
    return results


def print_results(
    results: dict[str, tuple[float, float, float]],
    confidence_level: float,
    n: int,
    dataset_name: str,
) -> None:
    """Print a formatted results table to stdout."""
    pct = int(confidence_level * 100)
    overall_keys = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "macro_precision",
        "macro_recall",
    ]

    header = f"  {'Metric':<24}  {'Value':>7}  {'CI Low':>7}  {'CI High':>7}"
    divider = f"  {'-' * 52}"

    print(f"\n{'=' * 60}")
    print(f"  Dataset : {dataset_name}  (N={n})")
    print(f"  CI      : BCa bootstrap {pct}%")
    print(f"{'=' * 60}")

    print("\n  Overall")
    print(header)
    print(divider)
    for key in overall_keys:
        if key in results:
            point, lo, hi = results[key]
            print(f"  {key:<24}  {point:>7.4f}  {lo:>7.4f}  {hi:>7.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute BCa bootstrap confidence intervals for classification metrics."
    )
    parser.add_argument(
        "csv_paths",
        nargs="+",
        metavar="CSV",
        help="One or more prediction CSV files in the tfu_predictions.csv format.",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=_DEFAULT_CONFIDENCE_LEVEL,
        help=f"Confidence level for the bootstrap interval (default: {_DEFAULT_CONFIDENCE_LEVEL}).",
    )
    parser.add_argument(
        "--n-resamples",
        type=int,
        default=_DEFAULT_N_RESAMPLES,
        help=f"Number of bootstrap resamples (default: {_DEFAULT_N_RESAMPLES}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    for csv_path in args.csv_paths:
        y_true, y_pred = load_predictions(csv_path)
        print(f"Loaded {len(y_true)} samples from {csv_path}.")
        results = compute_bootstrap_metrics(
            y_true,
            y_pred,
            n_resamples=args.n_resamples,
            confidence_level=args.confidence_level,
            random_state=args.seed,
        )
        print_results(results, args.confidence_level, len(y_true), Path(csv_path).name)


if __name__ == "__main__":
    main()
