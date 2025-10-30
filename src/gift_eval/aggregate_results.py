import argparse
import glob
import logging
from pathlib import Path
from typing import List

import pandas as pd

from src.gift_eval.constants import (
    ALL_DATASETS,
    DATASET_PROPERTIES,
    MED_LONG_DATASETS,
    PRETTY_NAMES,
)

logger = logging.getLogger(__name__)


def get_all_datasets_full_name() -> List[str]:
    """Get all possible dataset full names for validation."""
    terms = ["short", "medium", "long"]
    datasets_full_names: List[str] = []

    for name in ALL_DATASETS:
        for term in terms:
            if term in ["medium", "long"] and name not in MED_LONG_DATASETS.split():
                continue

            if "/" in name:
                ds_key, ds_freq = name.split("/")
                ds_key = ds_key.lower()
                ds_key = PRETTY_NAMES.get(ds_key, ds_key)
            else:
                ds_key = name.lower()
                ds_key = PRETTY_NAMES.get(ds_key, ds_key)
                ds_freq = DATASET_PROPERTIES[ds_key]["frequency"]

            datasets_full_names.append(f"{ds_key}/{ds_freq}/{term}")

    return datasets_full_names


def aggregate_results(
    result_root_dir: str | Path,
) -> pd.DataFrame | None:
    """Aggregate results from multiple CSV files.

    Returns the combined dataframe. Optionally saves to
    <result_root_dir>/all_results.csv
    """
    result_root_dir = Path(result_root_dir)

    logger.info(f"Aggregating results in: {result_root_dir}")

    # Find all CSV result files under the provided root directory
    # Results are written per-dataset as <result_root_dir>/<dataset_name>/results.csv
    result_files = glob.glob(f"{result_root_dir}/**/results.csv", recursive=True)

    if not result_files:
        logger.error("No result files found!")
        return None

    # Initialize empty list to store dataframes
    dataframes: List[pd.DataFrame] = []

    # Read and combine all CSV files
    for file in result_files:
        try:
            df = pd.read_csv(file)
            if len(df) > 0:
                dataframes.append(df)
            else:
                logger.warning(f"Empty file: {file}")
        except pd.errors.EmptyDataError:
            logger.warning(f"Skipping empty file: {file}")
        except Exception as e:
            logger.error(f"Error reading {file}: {str(e)}")

    if dataframes:
        # Combine all dataframes and sort by dataset
        combined_df = pd.concat(dataframes, ignore_index=True).sort_values("dataset")

        # Check for duplicates
        if len(combined_df) != len(set(combined_df.dataset)):
            duplicate_datasets = combined_df.dataset[
                combined_df.dataset.duplicated()
            ].tolist()
            logger.warning(f"Warning: Duplicate datasets found: {duplicate_datasets}")
            # Remove duplicates, keeping the first occurrence
            combined_df = combined_df.drop_duplicates(subset=["dataset"], keep="first")
            logger.info(
                f"Removed duplicates, {len(combined_df)} unique datasets remaining"
            )

        logger.info(f"Combined results: {len(combined_df)} datasets")
    else:
        logger.warning("No valid CSV files found to combine")
        return None

    # Get all expected datasets and compare with completed ones
    all_datasets_full_name = get_all_datasets_full_name()
    completed_experiments = combined_df.dataset.tolist()

    completed_experiments_clean = [
        exp for exp in completed_experiments if exp in all_datasets_full_name
    ]
    missing_or_failed_experiments = [
        exp for exp in all_datasets_full_name if exp not in completed_experiments_clean
    ]

    logger.info("=== EXPERIMENT SUMMARY ===")
    logger.info(f"Total expected datasets: {len(all_datasets_full_name)}")
    logger.info(f"Completed experiments: {len(completed_experiments_clean)}")
    logger.info(f"Missing/failed experiments: {len(missing_or_failed_experiments)}")

    logger.info("Completed experiments:")
    for i, exp in enumerate(completed_experiments_clean):
        logger.info(f"  {i + 1:3d}: {exp}")

    if missing_or_failed_experiments:
        logger.info("Missing or failed experiments:")
        for i, exp in enumerate(missing_or_failed_experiments):
            logger.info(f"  {i + 1:3d}: {exp}")

    # Calculate completion percentage
    completion_rate = (
        len(completed_experiments_clean) / len(all_datasets_full_name) * 100
    )
    logger.info(f"Completion rate: {completion_rate:.1f}%")

    # Save combined results
    output_file = result_root_dir / "all_results.csv"
    combined_df.to_csv(output_file, index=False)
    logger.info(f"Combined results saved to: {output_file}")

    return combined_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate GIFT-Eval results from multiple CSV files"
    )
    parser.add_argument(
        "--result_root_dir",
        type=str,
        required=True,
        help="Root directory containing result subdirectories",
    )

    args = parser.parse_args()
    args.result_root_dir = Path(args.result_root_dir)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger.info(f"Searching in directory: {args.result_root_dir}")

    aggregate_results(
        result_root_dir=args.result_root_dir,
    )
