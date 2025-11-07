"""Utilities for persisting and aggregating GIFT-Eval results."""

import argparse
import csv
import glob
import logging
from pathlib import Path

import pandas as pd

from src.gift_eval.constants import (
    ALL_DATASETS,
    DATASET_PROPERTIES,
    MED_LONG_DATASETS,
    PRETTY_NAMES,
    STANDARD_METRIC_NAMES,
)
from src.gift_eval.core import DatasetMetadata, EvaluationItem

logger = logging.getLogger(__name__)


def _ensure_results_csv(csv_file_path: Path) -> None:
    if not csv_file_path.exists():
        csv_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_file_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            header = (
                ["dataset", "model"]
                + [f"eval_metrics/{name}" for name in STANDARD_METRIC_NAMES]
                + ["domain", "num_variates"]
            )
            writer.writerow(header)


def write_results_to_disk(
    items: list[EvaluationItem],
    dataset_name: str,
    output_dir: Path,
    model_name: str,
    create_plots: bool,
) -> None:
    output_dir = output_dir / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv_path = output_dir / "results.csv"
    _ensure_results_csv(output_csv_path)

    try:
        import matplotlib.pyplot as plt  # Local import to avoid unnecessary dependency at module import time
    except ImportError:  # pragma: no cover - guard for optional dependency
        plt = None

    with open(output_csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for item in items:
            md: DatasetMetadata = item.dataset_metadata
            metric_values: list[float | None] = []
            for metric_name in STANDARD_METRIC_NAMES:
                value = item.metrics.get(metric_name, None)
                if value is None:
                    metric_values.append(None)
                else:
                    if hasattr(value, "__len__") and not isinstance(value, (str, bytes)) and len(value) == 1:
                        value = value[0]
                    elif hasattr(value, "item"):
                        value = value.item()
                    metric_values.append(value)

            ds_key = md.key.lower()
            props = DATASET_PROPERTIES.get(ds_key, {})
            domain = props.get("domain", "unknown")
            num_variates = props.get("num_variates", 1 if md.to_univariate else md.target_dim)

            row = [md.full_name, model_name] + metric_values + [domain, num_variates]
            writer.writerow(row)

            if create_plots and item.figures and plt is not None:
                plots_dir = output_dir / "plots" / md.key / md.term
                plots_dir.mkdir(parents=True, exist_ok=True)
                for fig, filename in item.figures:
                    filepath = plots_dir / filename
                    fig.savefig(filepath, dpi=300, bbox_inches="tight")
                    plt.close(fig)

    logger.info(
        "Evaluation complete for dataset '%s'. Results saved to %s",
        dataset_name,
        output_csv_path,
    )
    if create_plots:
        logger.info("Plots saved under %s", output_dir / "plots")


def get_all_datasets_full_name() -> list[str]:
    """Get all possible dataset full names for validation."""

    terms = ["short", "medium", "long"]
    datasets_full_names: list[str] = []

    for name in ALL_DATASETS:
        for term in terms:
            if term in ["medium", "long"] and name not in MED_LONG_DATASETS:
                continue

            if "/" in name:
                ds_key, ds_freq = name.split("/")
                ds_key = ds_key.lower()
                ds_key = PRETTY_NAMES.get(ds_key, ds_key)
            else:
                ds_key = name.lower()
                ds_key = PRETTY_NAMES.get(ds_key, ds_key)
                ds_freq = DATASET_PROPERTIES.get(ds_key, {}).get("frequency")

            datasets_full_names.append(f"{ds_key}/{ds_freq if ds_freq else 'unknown'}/{term}")

    return datasets_full_names


def aggregate_results(result_root_dir: str | Path) -> pd.DataFrame | None:
    """Aggregate results from multiple CSV files into a single dataframe."""

    result_root = Path(result_root_dir)

    logger.info("Aggregating results in: %s", result_root)

    result_files = glob.glob(f"{result_root}/**/results.csv", recursive=True)

    if not result_files:
        logger.error("No result files found!")
        return None

    dataframes: list[pd.DataFrame] = []
    for file in result_files:
        try:
            df = pd.read_csv(file)
            if len(df) > 0:
                dataframes.append(df)
            else:
                logger.warning("Empty file: %s", file)
        except pd.errors.EmptyDataError:
            logger.warning("Skipping empty file: %s", file)
        except Exception as exc:
            logger.error("Error reading %s: %s", file, exc)

    if not dataframes:
        logger.warning("No valid CSV files found to combine")
        return None

    combined_df = pd.concat(dataframes, ignore_index=True).sort_values("dataset")

    if len(combined_df) != len(set(combined_df.dataset)):
        duplicate_datasets = combined_df.dataset[combined_df.dataset.duplicated()].tolist()
        logger.warning("Warning: Duplicate datasets found: %s", duplicate_datasets)
        combined_df = combined_df.drop_duplicates(subset=["dataset"], keep="first")
        logger.info("Removed duplicates, %s unique datasets remaining", len(combined_df))

    logger.info("Combined results: %s datasets", len(combined_df))

    all_datasets_full_name = get_all_datasets_full_name()
    completed_experiments = combined_df.dataset.tolist()

    completed_experiments_clean = [exp for exp in completed_experiments if exp in all_datasets_full_name]
    missing_or_failed_experiments = [exp for exp in all_datasets_full_name if exp not in completed_experiments_clean]

    logger.info("=== EXPERIMENT SUMMARY ===")
    logger.info("Total expected datasets: %s", len(all_datasets_full_name))
    logger.info("Completed experiments: %s", len(completed_experiments_clean))
    logger.info("Missing/failed experiments: %s", len(missing_or_failed_experiments))

    logger.info("Completed experiments:")
    for idx, exp in enumerate(completed_experiments_clean, start=1):
        logger.info("  %3d: %s", idx, exp)

    if missing_or_failed_experiments:
        logger.info("Missing or failed experiments:")
        for idx, exp in enumerate(missing_or_failed_experiments, start=1):
            logger.info("  %3d: %s", idx, exp)

    completion_rate = (
        len(completed_experiments_clean) / len(all_datasets_full_name) * 100 if all_datasets_full_name else 0.0
    )
    logger.info("Completion rate: %.1f%%", completion_rate)

    output_file = result_root / "all_results.csv"
    combined_df.to_csv(output_file, index=False)
    logger.info("Combined results saved to: %s", output_file)

    return combined_df


__all__ = [
    "aggregate_results",
    "get_all_datasets_full_name",
    "write_results_to_disk",
]


def main() -> None:
    """CLI entry point for aggregating results from disk."""

    parser = argparse.ArgumentParser(description="Aggregate GIFT-Eval results from multiple CSV files")
    parser.add_argument(
        "--result_root_dir",
        type=str,
        required=True,
        help="Root directory containing result subdirectories",
    )

    args = parser.parse_args()
    result_root_dir = Path(args.result_root_dir)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.info("Searching in directory: %s", result_root_dir)

    aggregate_results(result_root_dir=result_root_dir)


if __name__ == "__main__":
    main()
