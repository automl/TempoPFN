"""Core data structures and helpers shared across GIFT-Eval modules."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from src.gift_eval.constants import ALL_DATASETS


@dataclass
class DatasetMetadata:
    """Structured description of a dataset/term combination."""

    full_name: str
    key: str
    freq: str
    term: str
    season_length: int
    target_dim: int
    to_univariate: bool
    prediction_length: int
    windows: int


@dataclass
class EvaluationItem:
    """Container for evaluation results and optional figures."""

    dataset_metadata: DatasetMetadata
    metrics: Dict
    figures: List[Tuple[object, str]]


DatasetSelection = Union[List[str], Tuple[str, ...], str]


def expand_datasets_arg(datasets: DatasetSelection) -> List[str]:
    """Normalize dataset selection strings to explicit lists."""

    if isinstance(datasets, str):
        dataset_list = [datasets]
    else:
        dataset_list = list(datasets)

    if not dataset_list:
        return []

    if dataset_list[0] == "all":
        return list(ALL_DATASETS)

    for dataset in dataset_list:
        if dataset not in ALL_DATASETS:
            raise ValueError(f"Invalid dataset: {dataset}. Use one of {ALL_DATASETS}")

    return dataset_list


__all__ = [
    "DatasetMetadata",
    "EvaluationItem",
    "DatasetSelection",
    "expand_datasets_arg",
]


