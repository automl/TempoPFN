#!/usr/bin/env python
"""
GIFT-Eval Runner Script

This script evaluates the Time Series model on GIFT-Eval datasets using the `src/gift_eval` pipeline.

- Uses `src/gift_eval/data.py` for dataset handling.
- Uses `src/gift_eval/predictor.TimeSeriesPredictor` for inference.
- Loads a model from a checkpoint.
- Writes per-dataset CSV metrics to `output_dir` without creating plots.
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional

from examples.utils import download_checkpoint_if_needed
from src.gift_eval.constants import ALL_DATASETS
from src.gift_eval.evaluate import evaluate_datasets
from src.gift_eval.predictor import TimeSeriesPredictor
from src.gift_eval.results import aggregate_results, write_results_to_disk


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
logger = logging.getLogger("gift_eval_runner")


def _expand_datasets_arg(datasets_arg: List[str] | str) -> List[str]:
    """Expand dataset argument to list of dataset names."""
    if isinstance(datasets_arg, str):
        if datasets_arg == "all":
            return list(ALL_DATASETS)
        datasets_list = [datasets_arg]
    else:
        datasets_list = datasets_arg
        if datasets_list and datasets_list[0] == "all":
            return list(ALL_DATASETS)

    for ds in datasets_list:
        if ds not in ALL_DATASETS:
            raise ValueError(f"Invalid dataset: {ds}. Use one of {ALL_DATASETS}")
    return datasets_list


def run_evaluation(
    predictor: TimeSeriesPredictor,
    datasets_arg: List[str] | str,
    terms_arg: List[str],
    dataset_storage_path: str,
    max_windows_arg: Optional[int],
    batch_size_arg: int,
    max_context_length_arg: Optional[int],
    output_dir_arg: str,
    model_name_arg: str,
    after_each_dataset_flush: bool = True,
) -> None:
    """Run evaluation on specified datasets."""
    datasets_to_run = _expand_datasets_arg(datasets_arg)
    results_root = Path(output_dir_arg)

    for ds_name in datasets_to_run:
        items = evaluate_datasets(
            predictor=predictor,
            dataset=ds_name,
            dataset_storage_path=dataset_storage_path,
            terms=terms_arg,
            max_windows=max_windows_arg,
            batch_size=batch_size_arg,
            max_context_length=max_context_length_arg,
            create_plots=False,
            max_plots_per_dataset=0,
        )
        write_results_to_disk(
            items=items,
            dataset_name=ds_name,
            output_dir=results_root,
            model_name=model_name_arg,
            create_plots=False,
        )
        if after_each_dataset_flush:
            logger.info("Flushed results for %s", ds_name)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="GIFT-Eval Runner: Evaluate TimeSeriesModel on GIFT-Eval datasets"
    )

    # Model configuration
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model checkpoint. If not provided, will download from checkpoint_url.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/example.yaml",
        help="Path to model config YAML (default: configs/example.yaml)",
    )
    parser.add_argument(
        "--checkpoint_url",
        type=str,
        default="https://www.dropbox.com/scl/fi/mqsni5lehooyaw93y3uzq/checkpoint_38M.pth?rlkey=3uyehvmtted02xkha24zgpzb6&st=seevsbkn&dl=0",
        help="URL to download checkpoint from if model_path is not provided",
    )
    parser.add_argument(
        "--download_dir",
        type=str,
        default="models",
        help="Directory to download checkpoint to (default: models)",
    )

    # Dataset configuration
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["all"],
        help='List of dataset names or ["all"] (default: all)',
    )
    parser.add_argument(
        "--terms",
        type=str,
        nargs="+",
        default=["short", "medium", "long"],
        help="Prediction terms to evaluate (default: short medium long)",
    )
    parser.add_argument(
        "--dataset_storage_path",
        type=str,
        default="/work/dlclarge2/moroshav-GiftEvalPretrain/gift_eval",
        # required=True,
        help="Path to the root of the gift eval datasets storage directory",
    )
    parser.add_argument(
        "--max_windows",
        type=int,
        default=20,
        help="Maximum number of windows to use for evaluation (default: 20)",
    )

    # Inference configuration
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for inference (default: 128)",
    )
    parser.add_argument(
        "--max_context_length",
        type=int,
        default=3072,
        help="Maximum context length (default: 3072)",
    )

    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="gift_eval_results",
        help="Output directory for results (default: gift_eval_results)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="TempoPFN",
        help="Model name identifier for results (default: TempoPFN)",
    )
    parser.add_argument(
        "--no_flush",
        action="store_true",
        help="Disable flushing results after each dataset",
    )

    args = parser.parse_args()

    # Resolve paths
    config_path = Path(args.config_path)
    download_dir = Path(args.download_dir)
    output_dir = Path(args.output_dir)

    # Determine model path
    resolved_model_path = None
    if args.model_path:
        resolved_model_path = args.model_path
    elif args.checkpoint_url:
        resolved_model_path = download_checkpoint_if_needed(
            args.checkpoint_url, target_dir=download_dir
        )

    if not resolved_model_path:
        raise FileNotFoundError(
            "No model checkpoint provided. Set --model_path or --checkpoint_url."
        )

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    logger.info("Loading predictor from checkpoint: %s", resolved_model_path)
    predictor = TimeSeriesPredictor.from_paths(
        model_path=resolved_model_path,
        config_path=str(config_path),
        ds_prediction_length=1,  # placeholder; set per dataset
        ds_freq="D",  # placeholder; set per dataset
        batch_size=args.batch_size,
        max_context_length=args.max_context_length,
    )

    logger.info("Starting evaluation...")
    logger.info("  Datasets: %s", args.datasets)
    logger.info("  Terms: %s", args.terms)
    logger.info("  Output directory: %s", output_dir)

    # Run evaluation
    run_evaluation(
        predictor=predictor,
        datasets_arg=args.datasets,
        terms_arg=args.terms,
        dataset_storage_path=args.dataset_storage_path,
        max_windows_arg=args.max_windows,
        batch_size_arg=args.batch_size,
        max_context_length_arg=args.max_context_length,
        output_dir_arg=str(output_dir),
        model_name_arg=args.model_name,
        after_each_dataset_flush=not args.no_flush,
    )

    logger.info("Evaluation complete. See results under: %s", output_dir)
    
    # Aggregate all results into a single CSV file
    logger.info("Aggregating results from all datasets...")
    combined_df = aggregate_results(result_root_dir=output_dir)
    
    if combined_df is not None:
        logger.info("Successfully created aggregated results file: %s/all_results.csv", output_dir)
    else:
        logger.warning("No results to aggregate. Check that evaluation completed successfully.")


if __name__ == "__main__":
    main()

