import logging
import random
from collections.abc import Iterator

import numpy as np
import pandas as pd
import torch

from src.data.batch_composer import BatchComposer, ComposedDataset
from src.data.containers import BatchTimeSeriesContainer
from src.data.frequency import parse_frequency
from src.gift_eval.constants import ALL_DATASETS
from src.gift_eval.data import Dataset as GiftEvalDataset

logger = logging.getLogger(__name__)


class GiftEvalDataLoader:
    """
    Data loader for GIFT-eval datasets, converting them to BatchTimeSeriesContainer format.
    Supports both training and validation modes.
    """

    TERMS = ["short", "medium", "long"]

    def __init__(
        self,
        mode: str = "train",
        batch_size: int = 32,
        device: torch.device | None = None,
        shuffle: bool = True,
        to_univariate: bool = False,
        max_context_length: int | None = None,
        max_windows: int = 20,
        skip_datasets_with_nans: bool = False,
        datasets_to_use: list[str] | None = None,
        dataset_storage_path: str | None = None,
    ):
        """
        Initialize GIFT-eval data loader.

        Args:
            mode: Either "train" or "validation"
            batch_size: Number of samples per batch
            device: Device to load data to
            shuffle: Whether to shuffle data
            to_univariate: Whether to convert multivariate data to multiple univariate series
            max_context_length: Optional maximum total window length (context + forecast) to prevent memory issues
            max_windows: Number of windows to use for training/validation
            skip_datasets_with_nans: Whether to skip datasets/series that contain NaN values
            datasets_to_use: Optional list of dataset names to use. If None, uses all available datasets
            dataset_storage_path: Path on disk where GIFT-eval HuggingFace datasets are stored
        """
        # Use specified datasets or all available datasets if none specified
        if datasets_to_use is not None and len(datasets_to_use) > 0:
            # Validate that requested datasets are available
            invalid_datasets = [ds for ds in datasets_to_use if ds not in ALL_DATASETS]
            if invalid_datasets:
                logger.warning(f"Invalid datasets requested: {invalid_datasets}")
                logger.warning(f"Available datasets: {ALL_DATASETS}")
                # Use only valid datasets
                self.dataset_names = [ds for ds in datasets_to_use if ds in ALL_DATASETS]
            else:
                self.dataset_names = datasets_to_use
        else:
            self.dataset_names = ALL_DATASETS

        # Log dataset selection
        if datasets_to_use is not None and len(datasets_to_use) > 0:
            logger.info(f"Using subset of datasets: {len(self.dataset_names)}/{len(ALL_DATASETS)} datasets")
            logger.info(f"Selected datasets: {self.dataset_names}")
        else:
            logger.info(f"Using all available datasets: {len(self.dataset_names)} datasets")

        self.terms = self.TERMS
        self.mode = mode
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.to_univariate = to_univariate
        self.max_context_length = max_context_length
        self.skip_datasets_with_nans = skip_datasets_with_nans

        # Window configuration based on mode
        self.max_windows = max_windows
        self.dataset_storage_path = dataset_storage_path

        # Load all datasets and prepare data
        self._load_datasets()

        # Create iterator state
        self._current_idx = 0
        self._epoch_data = []
        self._prepare_epoch_data()

    def _load_datasets(self) -> None:
        """Load all specified GIFT-eval datasets."""
        self.datasets = {}
        self.dataset_prediction_lengths = {}

        for dataset_name in self.dataset_names:
            if dataset_name.startswith("m4_"):
                max_windows = 1
            else:
                max_windows = self.max_windows
            try:
                # Determine if we need univariate conversion
                # First check with multivariate to see target dimension
                temp_dataset = GiftEvalDataset(
                    name=dataset_name,
                    term=self.terms[0],  # Use first term to check dimensionality
                    to_univariate=False,
                    max_windows=max_windows,
                    storage_path=self.dataset_storage_path,
                )

                # Convert to univariate if needed
                to_univariate = self.to_univariate and temp_dataset.target_dim > 1

                # Load datasets for all terms
                for term in self.terms:
                    dataset_key = f"{dataset_name}_{term}"
                    dataset = GiftEvalDataset(
                        name=dataset_name,
                        term=term,
                        to_univariate=to_univariate,
                        max_windows=max_windows,
                        storage_path=self.dataset_storage_path,
                    )

                    self.datasets[dataset_key] = dataset
                    self.dataset_prediction_lengths[dataset_key] = dataset.prediction_length

                    logger.info(
                        f"Loaded {dataset_key} - prediction_length: {dataset.prediction_length}, "
                        f"frequency: {dataset.freq}, target_dim: {dataset.target_dim}, "
                        f"min_length: {dataset._min_series_length}, windows: {dataset.windows}"
                    )

            except Exception as e:
                logger.warning(f"Failed to load dataset {dataset_name}: {str(e)}")
                continue

    def _contains_nan(self, data_entry: dict) -> bool:
        """Check if a data entry contains NaN values."""
        target = data_entry.get("target")
        if target is None:
            return False

        # Convert to numeric numpy array for robust NaN checking
        try:
            target_np = np.asarray(target, dtype=np.float32)
            return np.isnan(target_np).any()
        except Exception:
            logger.warning("NaN check: failed to coerce target to float32; skipping entry")
            return True

    def _convert_to_container(
        self, data_entries: list[dict], prediction_length: int, dataset_freq: str
    ) -> BatchTimeSeriesContainer:
        """Convert a batch of data entries to BatchTimeSeriesContainer format with fixed future length."""
        batch_size = len(data_entries)
        max_history_len = 0

        # First pass: determine max history length after truncation
        for entry in data_entries:
            target = np.asarray(entry["target"], dtype=np.float32)
            if target.ndim == 1:
                target = target.reshape(1, -1)

            _, seq_len = target.shape

            # Only consider up to the last (max_context_length) values
            effective_max_context = self.max_context_length if self.max_context_length is not None else seq_len
            if seq_len > effective_max_context:
                seq_len = effective_max_context

            # History is up to (max_context_length - prediction_length)
            history_len = max(0, min(seq_len, effective_max_context) - prediction_length)
            max_history_len = max(max_history_len, history_len)

        # Get number of channels from first entry
        first_target = np.asarray(data_entries[0]["target"], dtype=np.float32)
        if first_target.ndim == 1:
            # Shape to [channels, time]
            first_target = first_target.reshape(1, -1)
        num_channels = first_target.shape[0]

        # Allocate arrays
        history_values = np.full((batch_size, max_history_len, num_channels), np.nan, dtype=np.float32)
        future_values = np.full((batch_size, prediction_length, num_channels), np.nan, dtype=np.float32)
        history_mask = np.zeros((batch_size, max_history_len), dtype=bool)

        # Second pass: fill arrays
        for i, entry in enumerate(data_entries):
            target = np.asarray(entry["target"], dtype=np.float32)
            if target.ndim == 1:
                target = target.reshape(1, -1)

            # Truncate to last effective_max_context points if needed
            full_seq_len = target.shape[1]
            total_len_allowed = self.max_context_length if self.max_context_length is not None else full_seq_len
            total_len_for_entry = min(full_seq_len, total_len_allowed)

            if total_len_for_entry < prediction_length + 1:
                # Not enough length to build (history + future). Signal to caller.
                raise ValueError("Entry too short after max_context_length truncation to form history+future window")

            truncated = target[:, -total_len_for_entry:]
            cur_history_len = total_len_for_entry - prediction_length

            hist = truncated[:, :cur_history_len]  # [C, H]
            fut = truncated[:, cur_history_len : cur_history_len + prediction_length]  # [C, P]

            # Write into batch arrays with time last -> transpose to [H, C] / [P, C]
            history_values[i, :cur_history_len, :] = hist.T
            future_values[i, :, :] = fut.T
            history_mask[i, :cur_history_len] = True

        # Get start timestamp and frequency (replicate across batch)
        start_timestamp = data_entries[0]["start"]
        if hasattr(start_timestamp, "to_timestamp"):
            start_numpy = start_timestamp.to_timestamp().to_numpy()
        else:
            start_numpy = pd.Timestamp(start_timestamp).to_numpy()
        start_list = [start_numpy for _ in range(batch_size)]

        # Get frequency enum and replicate across batch
        frequency_enum = parse_frequency(dataset_freq)
        frequency_list = [frequency_enum for _ in range(batch_size)]

        # Create the container
        return BatchTimeSeriesContainer(
            history_values=torch.tensor(history_values, dtype=torch.float32),
            future_values=torch.tensor(future_values, dtype=torch.float32),
            start=start_list,
            frequency=frequency_list,
            history_mask=torch.tensor(history_mask, dtype=torch.bool) if self.mode == "train" else None,
        )

    def _prepare_epoch_data(self) -> None:
        """Prepare all batches for one epoch."""
        self._epoch_data = []

        for dataset_key, dataset in self.datasets.items():
            try:
                # Get appropriate dataset based on mode
                if self.mode == "train":
                    data = dataset.training_dataset
                else:
                    data = dataset.validation_dataset

                # Collect all valid data entries
                valid_entries = []
                dataset_freq = dataset.freq
                prediction_length = self.dataset_prediction_lengths[dataset_key]

                for entry in data:
                    # Skip if contains NaN and configured to do so
                    if self.skip_datasets_with_nans and self._contains_nan(entry):
                        continue

                    # Check if we have enough data
                    target = np.asarray(entry["target"])
                    if target.ndim == 1:
                        seq_len = len(target)
                    else:
                        seq_len = target.shape[1]

                    # Need at least prediction_length + 1 for training
                    if self.mode == "train" and seq_len < prediction_length + 1:
                        continue

                    valid_entries.append(entry)

                if not valid_entries:
                    logger.warning(f"No valid entries found for {dataset_key}")
                    continue

                # Create batches
                for i in range(0, len(valid_entries), self.batch_size):
                    batch_entries = valid_entries[i : i + self.batch_size]
                    try:
                        batch_container = self._convert_to_container(batch_entries, prediction_length, dataset_freq)
                        self._epoch_data.append((dataset_key, batch_container))
                    except Exception as e:
                        logger.warning(f"Failed to create batch for {dataset_key}: {str(e)}")
                        continue

            except Exception as e:
                logger.warning(
                    f"Failed to process dataset {dataset_key}: {str(e)}. "
                    f"Dataset may be too short for the required offset."
                )
                continue

        # Shuffle if in training mode
        if self.mode == "train" and self.shuffle:
            random.shuffle(self._epoch_data)

        logger.info(f"Prepared {len(self._epoch_data)} batches for {self.mode} mode")

    def __iter__(self) -> Iterator[BatchTimeSeriesContainer]:
        """Iterate through batches for one epoch."""
        # Reset index at the start of each epoch
        self._current_idx = 0

        # Reshuffle data for each new epoch if in training mode
        if self.mode == "train" and self.shuffle:
            random.shuffle(self._epoch_data)

        return self

    def __next__(self) -> BatchTimeSeriesContainer:
        """Get next batch."""
        if not self._epoch_data:
            raise StopIteration("No valid data available")

        # Check if we've exhausted the epoch
        if self._current_idx >= len(self._epoch_data):
            raise StopIteration

        # Get current batch
        dataset_key, batch = self._epoch_data[self._current_idx]
        self._current_idx += 1

        # Move to device if specified
        if self.device is not None:
            batch.to_device(self.device)

        return batch

    def __len__(self) -> int:
        """Return number of batches per epoch."""
        return len(self._epoch_data)


class CyclicGiftEvalDataLoader:
    """
    Wrapper for GiftEvalDataLoader that provides cycling behavior for training.
    This allows training for a fixed number of iterations per epoch, cycling through
    the available data as needed.
    """

    def __init__(self, base_loader: GiftEvalDataLoader, num_iterations_per_epoch: int):
        """
        Initialize the cyclic data loader.

        Args:
            base_loader: The underlying GiftEvalDataLoader
            num_iterations_per_epoch: Number of iterations to run per epoch
        """
        self.base_loader = base_loader
        self.num_iterations_per_epoch = num_iterations_per_epoch
        self.dataset_names = base_loader.dataset_names
        self.device = base_loader.device

    def __iter__(self) -> Iterator[BatchTimeSeriesContainer]:
        """Iterate for exactly num_iterations_per_epoch iterations."""
        self._current_iteration = 0
        self._base_iter = iter(self.base_loader)
        return self

    def __next__(self) -> BatchTimeSeriesContainer:
        """Get next batch, cycling through base loader as needed."""
        if self._current_iteration >= self.num_iterations_per_epoch:
            raise StopIteration

        try:
            batch = next(self._base_iter)
        except StopIteration:
            # Restart the base iterator when exhausted
            self._base_iter = iter(self.base_loader)
            batch = next(self._base_iter)

        self._current_iteration += 1
        return batch

    def __len__(self) -> int:
        """Return the configured number of iterations per epoch."""
        return self.num_iterations_per_epoch


def create_synthetic_dataloader(
    base_data_dir: str,
    batch_size: int = 128,
    num_batches_per_epoch: int = 1000,
    generator_proportions: dict[str, float] | None = None,
    mixed_batches: bool = True,
    augmentations: dict[str, bool] | None = None,
    augmentation_probabilities: dict[str, float] | None = None,
    device: torch.device | None = None,
    num_workers: int = 0,
    pin_memory: bool = True,
    global_seed: int = 42,
    nan_stats_path: str | None = None,
    nan_patterns_path: str | None = None,
    chosen_scaler_name: str | None = None,
) -> torch.utils.data.DataLoader:
    """
    Create a PyTorch DataLoader for training with saved generator batches.

    Args:
        base_data_dir: Base directory containing generator subdirectories
        batch_size: Size of each training batch
        num_batches_per_epoch: Number of batches per epoch
        generator_proportions: Dict mapping generator names to proportions
        mixed_batches: Whether to create mixed or uniform batches
        augmentations: Dict mapping augmentation names to booleans
        augmentation_probabilities: Dict mapping augmentation names to probabilities
        device: Target device
        num_workers: Number of DataLoader workers
        pin_memory: Whether to pin memory
        global_seed: Global random seed
        nan_stats_path: Path to nan stats file
        chosen_scaler_name: Name of the scaler that used in training

    Returns:
        PyTorch DataLoader
    """

    # Create batch composer
    composer = BatchComposer(
        base_data_dir=base_data_dir,
        generator_proportions=generator_proportions,
        mixed_batches=mixed_batches,
        device=device,
        augmentations=augmentations,
        augmentation_probabilities=augmentation_probabilities,
        global_seed=global_seed,
        nan_stats_path=nan_stats_path,
        nan_patterns_path=nan_patterns_path,
        chosen_scaler_name=chosen_scaler_name,
    )

    # Create dataset
    dataset = ComposedDataset(
        batch_composer=composer,
        num_batches_per_epoch=num_batches_per_epoch,
        batch_size=batch_size,
    )

    # Custom collate function for BatchTimeSeriesContainer
    def collate_fn(batch):
        """Custom collate function that returns a single BatchTimeSeriesContainer."""
        # Since each item is already a BatchTimeSeriesContainer with batch_size samples,
        # and DataLoader batch_size=1, we just return the first (and only) item
        return batch[0]

    # Create DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,  # Each dataset item is already a complete batch
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False,
    )

    logger.info(
        f"Created DataLoader with {len(dataset)} batches per epoch, "
        f"batch_size={batch_size}, mixed_batches={mixed_batches}"
    )

    return dataloader


class SyntheticValidationDataset(torch.utils.data.Dataset):
    """
    Fixed synthetic validation dataset that generates a small number of batches
    using the same composition approach as training data.
    """

    def __init__(
        self,
        base_data_dir: str,
        batch_size: int = 128,
        num_batches: int = 2,
        future_length: int = 512,
        generator_proportions: dict[str, float] | None = None,
        augmentations: dict[str, bool] | None = None,
        augmentation_probabilities: dict[str, float] | None = None,
        device: torch.device | None = None,
        global_seed: int = 42,
        chosen_scaler_name: str | None = None,
        nan_stats_path: str | None = None,
        nan_patterns_path: str | None = None,
        rank: int = 0,
        world_size: int = 1,
    ):
        """
        Initialize the validation dataset.

        Args:
            base_data_dir: Base directory containing generator subdirectories
            batch_size: Size of each validation batch
            num_batches: Number of validation batches to generate (1 or 2)
            generator_proportions: Dict mapping generator names to proportions
            device: Device to load tensors to
            global_seed: Global random seed
            chosen_scaler_name: Name of the scaler that used in training
        """
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.device = device

        # Create batch composer; force validation to use max-length windows (no length shortening)
        val_augmentations = dict(augmentations or {})
        val_augmentations["length_shortening"] = False

        self.batch_composer = BatchComposer(
            base_data_dir=base_data_dir,
            generator_proportions=generator_proportions,
            mixed_batches=True,  # Use mixed batches for validation
            device=device,
            global_seed=global_seed + 999999,
            augmentations=val_augmentations,
            augmentation_probabilities=augmentation_probabilities,
            nan_stats_path=nan_stats_path,
            nan_patterns_path=nan_patterns_path,
            chosen_scaler_name=chosen_scaler_name,
            rank=rank,
            world_size=world_size,
        )

        # Pre-generate fixed validation batches
        self.validation_batches = []
        for i in range(num_batches):
            batch, _ = self.batch_composer.create_batch(
                batch_size=batch_size,
                future_length=future_length,
                seed=global_seed + 999999 + i,  # Fixed seeds for reproducible validation
            )
            self.validation_batches.append(batch)

        logger.info(f"Created {num_batches} fixed validation batches with batch_size={batch_size}")

    def __len__(self) -> int:
        return self.num_batches

    def __getitem__(self, idx: int) -> BatchTimeSeriesContainer:
        """
        Get a pre-generated validation batch by index.

        Args:
            idx: Batch index

        Returns:
            BatchTimeSeriesContainer
        """
        if idx >= len(self.validation_batches):
            raise IndexError(f"Batch index {idx} out of range")

        batch = self.validation_batches[idx]

        # Move to device if needed
        if self.device is not None:
            batch.to_device(self.device)

        return batch


def create_synthetic_dataset(
    base_data_dir: str,
    batch_size: int = 128,
    num_batches_per_epoch: int = 1000,
    generator_proportions: dict[str, float] | None = None,
    mixed_batches: bool = True,
    augmentations: dict[str, bool] | None = None,
    augmentation_probabilities: dict[str, float] | None = None,
    global_seed: int = 42,
    nan_stats_path: str | None = None,
    nan_patterns_path: str | None = None,
    chosen_scaler_name: str | None = None,
    rank: int = 0,
    world_size: int = 1,
) -> ComposedDataset:
    """
    Creates the ComposedDataset for training with saved generator batches.

    Args:
        base_data_dir: Base directory containing generator subdirectories.
        batch_size: Size of each training batch.
        num_batches_per_epoch: Number of batches per epoch.
        generator_proportions: Dict mapping generator names to proportions.
        mixed_batches: Whether to create mixed or uniform batches.
        augmentations: Dict mapping augmentation names to booleans.
        global_seed: Global random seed.
        nan_stats_path: Path to nan stats file.
        chosen_scaler_name: Name of the scaler to use.
    Returns:
        A ComposedDataset instance.
    """
    # Create batch composer
    composer = BatchComposer(
        base_data_dir=base_data_dir,
        generator_proportions=generator_proportions,
        mixed_batches=mixed_batches,
        device=None,  # Device is handled in the training loop
        augmentations=augmentations,
        augmentation_probabilities=augmentation_probabilities,
        global_seed=global_seed,
        nan_stats_path=nan_stats_path,
        nan_patterns_path=nan_patterns_path,
        chosen_scaler_name=chosen_scaler_name,
        rank=rank,
        world_size=world_size,
    )

    # Create and return the dataset
    dataset = ComposedDataset(
        batch_composer=composer,
        num_batches_per_epoch=num_batches_per_epoch,
        batch_size=batch_size,
    )

    logger.info(
        f"Created ComposedDataset with {len(dataset)} batches per epoch, "
        f"batch_size={batch_size}, mixed_batches={mixed_batches}"
    )

    return dataset
