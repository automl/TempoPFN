import argparse
import logging
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
from gluonts.model.evaluation import evaluate_model
from gluonts.time_feature import get_seasonality
from linear_operator.utils.cholesky import NumericalWarning

from src.gift_eval.constants import (
    DATASET_PROPERTIES,
    MED_LONG_DATASETS,
    METRICS,
    PRETTY_NAMES,
)
from src.gift_eval.core import DatasetMetadata, EvaluationItem, expand_datasets_arg
from src.gift_eval.data import Dataset
from src.gift_eval.predictor import TimeSeriesPredictor
from src.gift_eval.results import write_results_to_disk
from src.plotting.gift_eval_utils import create_plots_for_dataset

logger = logging.getLogger(__name__)

# Warnings configuration
warnings.filterwarnings("ignore", category=NumericalWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
matplotlib.set_loglevel("WARNING")
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)


class WarningFilter(logging.Filter):
    def __init__(self, text_to_filter: str) -> None:
        super().__init__()
        self.text_to_filter = text_to_filter

    def filter(self, record: logging.LogRecord) -> bool:
        return self.text_to_filter not in record.getMessage()


# Filter out gluonts warnings about mean predictions
gts_logger = logging.getLogger("gluonts.model.forecast")
gts_logger.addFilter(
    WarningFilter("The mean prediction is not stored in the forecast data")
)


def construct_evaluation_data(
    dataset_name: str,
    dataset_storage_path: str,
    terms: List[str] = ["short", "medium", "long"],
    max_windows: Optional[int] = None,
) -> List[Tuple[Dataset, DatasetMetadata]]:
    """Build datasets and rich metadata per term for a dataset name."""
    sub_datasets: List[Tuple[Dataset, DatasetMetadata]] = []

    if "/" in dataset_name:
        ds_key, ds_freq = dataset_name.split("/")
        ds_key = ds_key.lower()
        ds_key = PRETTY_NAMES.get(ds_key, ds_key)
    else:
        ds_key = dataset_name.lower()
        ds_key = PRETTY_NAMES.get(ds_key, ds_key)
        ds_freq = DATASET_PROPERTIES.get(ds_key, {}).get("frequency")

    for term in terms:
        # Skip medium/long terms for datasets that don't support them
        if (
            term == "medium" or term == "long"
        ) and dataset_name not in MED_LONG_DATASETS:
            continue

        # Probe once to determine dimensionality
        probe_dataset = Dataset(
            name=dataset_name,
            term=term,
            to_univariate=False,
            storage_path=dataset_storage_path,
            max_windows=max_windows,
        )

        to_univariate = probe_dataset.target_dim > 1

        dataset = Dataset(
            name=dataset_name,
            term=term,
            to_univariate=to_univariate,
            storage_path=dataset_storage_path,
            max_windows=max_windows,
        )

        # Compute metadata
        season_length = get_seasonality(dataset.freq)
        actual_freq = ds_freq if ds_freq else dataset.freq
        
        metadata = DatasetMetadata(
            full_name=f"{ds_key}/{actual_freq}/{term}",
            key=ds_key,
            freq=actual_freq,
            term=term,
            season_length=season_length,
            target_dim=probe_dataset.target_dim,
            to_univariate=to_univariate,
            prediction_length=dataset.prediction_length,
            windows=dataset.windows,
        )

        sub_datasets.append((dataset, metadata))

    return sub_datasets


def evaluate_datasets(
    predictor: TimeSeriesPredictor,
    dataset: str,
    dataset_storage_path: str,
    terms: List[str] = ["short", "medium", "long"],
    max_windows: Optional[int] = None,
    batch_size: int = 48,
    max_context_length: Optional[int] = 1024,
    create_plots: bool = False,
    max_plots_per_dataset: int = 10,
) -> List[EvaluationItem]:
    """Evaluate predictor on one dataset across the requested terms."""
    sub_datasets = construct_evaluation_data(
        dataset_name=dataset,
        dataset_storage_path=dataset_storage_path,
        terms=terms,
        max_windows=max_windows,
    )

    results: List[EvaluationItem] = []
    for i, (sub_dataset, metadata) in enumerate(sub_datasets):
        logger.info(f"Evaluating {i + 1}/{len(sub_datasets)}: {metadata.full_name}")
        logger.info(f"  Dataset size: {len(sub_dataset.test_data)}")
        logger.info(f"  Frequency: {sub_dataset.freq}")
        logger.info(f"  Term: {metadata.term}")
        logger.info(f"  Prediction length: {sub_dataset.prediction_length}")
        logger.info(f"  Target dimensions: {sub_dataset.target_dim}")
        logger.info(f"  Windows: {sub_dataset.windows}")

        # Update context on the reusable predictor
        predictor.set_dataset_context(
            prediction_length=sub_dataset.prediction_length,
            freq=sub_dataset.freq,
            batch_size=batch_size,
            max_context_length=max_context_length,
        )

        res = evaluate_model(
            model=predictor,
            test_data=sub_dataset.test_data,
            metrics=METRICS,
            axis=None,
            mask_invalid_label=True,
            allow_nan_forecast=False,
            seasonality=metadata.season_length,
        )

        figs: List[Tuple[object, str]] = []
        if create_plots:
            forecasts = predictor.predict(sub_dataset.test_data.input)
            figs = create_plots_for_dataset(
                forecasts=forecasts,
                test_data=sub_dataset.test_data,
                dataset_metadata=metadata,
                max_plots=max_plots_per_dataset,
                max_context_length=max_context_length,
            )

        results.append(
            EvaluationItem(dataset_metadata=metadata, metrics=res, figures=figs)
        )

    return results


def _run_evaluation(
    predictor: TimeSeriesPredictor,
    datasets: List[str] | str,
    terms: List[str],
    dataset_storage_path: str,
    max_windows: Optional[int] = None,
    batch_size: int = 48,
    max_context_length: Optional[int] = 1024,
    output_dir: str = "gift_eval_results",
    model_name: str = "TimeSeriesModel",
    create_plots: bool = False,
    max_plots: int = 10,
) -> None:
    """Shared evaluation workflow used by both entry points."""
    datasets_to_run = expand_datasets_arg(datasets)
    results_root = Path(output_dir)

    for ds_name in datasets_to_run:
        items = evaluate_datasets(
            predictor=predictor,
            dataset=ds_name,
            dataset_storage_path=dataset_storage_path,
            terms=terms,
            max_windows=max_windows,
            batch_size=batch_size,
            max_context_length=max_context_length,
            create_plots=create_plots,
            max_plots_per_dataset=max_plots,
        )
        write_results_to_disk(
            items=items,
            dataset_name=ds_name,
            output_dir=results_root,
            model_name=model_name,
            create_plots=create_plots,
        )


def evaluate_from_paths(
    model_path: str,
    config_path: str,
    datasets: List[str] | str,
    terms: List[str],
    dataset_storage_path: str,
    max_windows: Optional[int] = None,
    batch_size: int = 48,
    max_context_length: Optional[int] = 1024,
    output_dir: str = "gift_eval_results",
    model_name: str = "TimeSeriesModel",
    create_plots: bool = False,
    max_plots: int = 10,
) -> None:
    """Entry point: load model from disk and save metrics/plots to disk."""
    # Validate inputs early
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config path does not exist: {config_path}")

    predictor = TimeSeriesPredictor.from_paths(
        model_path=model_path,
        config_path=config_path,
        ds_prediction_length=1,  # placeholder; set per dataset below
        ds_freq="D",  # placeholder; set per dataset below
        batch_size=batch_size,
        max_context_length=max_context_length,
    )

    _run_evaluation(
        predictor=predictor,
        datasets=datasets,
        terms=terms,
        dataset_storage_path=dataset_storage_path,
        max_windows=max_windows,
        batch_size=batch_size,
        max_context_length=max_context_length,
        output_dir=output_dir,
        model_name=model_name,
        create_plots=create_plots,
        max_plots=max_plots,
    )


def evaluate_in_memory(
    model,
    config: dict,
    datasets: List[str] | str,
    terms: List[str],
    dataset_storage_path: str,
    max_windows: Optional[int] = None,
    batch_size: int = 48,
    max_context_length: Optional[int] = 1024,
    output_dir: str = "gift_eval_results",
    model_name: str = "TimeSeriesModel",
    create_plots: bool = False,
    max_plots: int = 10,
) -> None:
    """Entry point: evaluate in-memory model and return results per dataset."""
    predictor = TimeSeriesPredictor.from_model(
        model=model,
        config=config,
        ds_prediction_length=1,  # placeholder; set per dataset below
        ds_freq="D",  # placeholder; set per dataset below
        batch_size=batch_size,
        max_context_length=max_context_length,
    )

    _run_evaluation(
        predictor=predictor,
        datasets=datasets,
        terms=terms,
        dataset_storage_path=dataset_storage_path,
        max_windows=max_windows,
        batch_size=batch_size,
        max_context_length=max_context_length,
        output_dir=output_dir,
        model_name=model_name,
        create_plots=create_plots,
        max_plots=max_plots,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate TimeSeriesModel on GIFT-Eval datasets"
    )

    # Model configuration
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the model configuration YAML file",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="TimeSeriesModel",
        help="Name identifier for the model",
    )

    # Dataset configuration
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        help="Comma-separated list of dataset names to evaluate (or 'all')",
    )
    parser.add_argument(
        "--dataset_storage_path",
        type=str,
        default="/work/dlclarge2/moroshav-GiftEvalPretrain/gift_eval",
        help="Path to the dataset storage directory (default: GIFT_EVAL)",
    )
    parser.add_argument(
        "--terms",
        type=str,
        default="short,medium,long",
        help="Comma-separated list of prediction terms to evaluate",
    )
    parser.add_argument(
        "--max_windows",
        type=int,
        default=None,
        help="Maximum number of windows to use for evaluation",
    )

    # Inference configuration
    parser.add_argument(
        "--batch_size", type=int, default=48, help="Batch size for model inference"
    )
    parser.add_argument(
        "--max_context_length",
        type=int,
        default=1024,
        help="Maximum context length to use (None for no limit)",
    )

    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="gift_eval_results",
        help="Directory to save evaluation results",
    )

    # Plotting configuration
    parser.add_argument(
        "--create_plots",
        action="store_true",
        help="Create and save plots for each evaluation window",
    )
    parser.add_argument(
        "--max_plots_per_dataset",
        type=int,
        default=10,
        help="Maximum number of plots to create per dataset term",
    )

    args = parser.parse_args()
    args.terms = args.terms.split(",")
    args.datasets = args.datasets.split(",")
    return args


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


if __name__ == "__main__":
    _configure_logging()
    args = _parse_args()
    logger.info(f"Command Line Arguments: {vars(args)}")
    try:
        evaluate_from_paths(
            model_path=args.model_path,
            config_path=args.config_path,
            datasets=args.datasets,
            terms=args.terms,
            dataset_storage_path=args.dataset_storage_path,
            max_windows=args.max_windows,
            batch_size=args.batch_size,
            max_context_length=args.max_context_length,
            output_dir=args.output_dir,
            model_name=args.model_name,
            create_plots=args.create_plots,
            max_plots=args.max_plots_per_dataset,
        )
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise
