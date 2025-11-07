import json
import logging
import random

import numpy as np
import pandas as pd
import torch

from src.data.augmentations import (
    NanAugmenter,
)
from src.data.constants import DEFAULT_NAN_STATS_PATH, LENGTH_CHOICES, LENGTH_WEIGHTS
from src.data.containers import BatchTimeSeriesContainer
from src.data.datasets import CyclicalBatchDataset
from src.data.frequency import Frequency
from src.data.scalers import MeanScaler, MedianScaler, MinMaxScaler, RobustScaler
from src.data.utils import sample_future_length

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BatchComposer:
    """
    Composes batches from saved generator data according to specified proportions.
    Manages multiple CyclicalBatchDataset instances and creates uniform or mixed batches.
    """

    def __init__(
        self,
        base_data_dir: str,
        generator_proportions: dict[str, float] | None = None,
        mixed_batches: bool = True,
        device: torch.device | None = None,
        augmentations: dict[str, bool] | None = None,
        augmentation_probabilities: dict[str, float] | None = None,
        nan_stats_path: str | None = None,
        nan_patterns_path: str | None = None,
        global_seed: int = 42,
        chosen_scaler_name: str | None = None,
        rank: int = 0,
        world_size: int = 1,
    ):
        """
        Initialize the BatchComposer.

        Args:
            base_data_dir: Base directory containing generator subdirectories
            generator_proportions: Dict mapping generator names to proportions
            mixed_batches: If True, create mixed batches; if False, uniform batches
            device: Device to load tensors to
            augmentations: Dict mapping augmentation names to booleans
            augmentation_probabilities: Dict mapping augmentation names to probabilities
            global_seed: Global random seed
            chosen_scaler_name: Name of the scaler that used in training
            rank: Rank of current process for distributed data loading
            world_size: Total number of processes for distributed data loading
        """
        self.base_data_dir = base_data_dir
        self.mixed_batches = mixed_batches
        self.device = device
        self.global_seed = global_seed
        self.nan_stats_path = nan_stats_path
        self.nan_patterns_path = nan_patterns_path
        self.rank = rank
        self.world_size = world_size
        self.augmentation_probabilities = augmentation_probabilities or {
            "noise_augmentation": 0.3,
            "scaler_augmentation": 0.5,
        }
        # Optional preferred scaler name provided by training config
        self.chosen_scaler_name = chosen_scaler_name.lower() if chosen_scaler_name is not None else None

        # Setup random state
        self.rng = np.random.default_rng(global_seed)
        random.seed(global_seed)
        torch.manual_seed(global_seed)

        # Setup augmentations
        self._setup_augmentations(augmentations)

        # Setup generator proportions
        self._setup_proportions(generator_proportions)

        # Initialize datasets
        self.datasets = self._initialize_datasets()

        logger.info(
            f"Initialized BatchComposer with {len(self.datasets)} generators, "
            f"mixed_batches={mixed_batches}, proportions={self.generator_proportions}, "
            f"augmentations={self.augmentations}, "
            f"augmentation_probabilities={self.augmentation_probabilities}"
        )

    def _setup_augmentations(self, augmentations: dict[str, bool] | None):
        """Setup only the augmentations that should remain online (NaN)."""
        default_augmentations = {
            "nan_augmentation": False,
            "scaler_augmentation": False,
            "length_shortening": False,
        }

        self.augmentations = augmentations or default_augmentations

        # Initialize NaN augmenter if needed
        self.nan_augmenter = None
        if self.augmentations.get("nan_augmentation", False):
            stats_path_to_use = self.nan_stats_path or DEFAULT_NAN_STATS_PATH
            stats = json.load(open(stats_path_to_use))
            self.nan_augmenter = NanAugmenter(
                p_series_has_nan=stats["p_series_has_nan"],
                nan_ratio_distribution=stats["nan_ratio_distribution"],
                nan_length_distribution=stats["nan_length_distribution"],
                nan_patterns_path=self.nan_patterns_path,
            )

    def _should_apply_scaler_augmentation(self) -> bool:
        """
        Decide whether to apply scaler augmentation for a single series based on
        the boolean toggle and probability from the configuration.
        """
        if not self.augmentations.get("scaler_augmentation", False):
            return False
        probability = float(self.augmentation_probabilities.get("scaler_augmentation", 0.0))
        probability = max(0.0, min(1.0, probability))
        return bool(self.rng.random() < probability)

    def _choose_random_scaler(self) -> object | None:
        """
        Choose a random scaler for augmentation, explicitly avoiding the one that
        is already selected in the training configuration (if any).

        Returns an instance of the selected scaler or None when no valid option exists.
        """
        chosen: str | None = None
        if self.chosen_scaler_name is not None:
            chosen = self.chosen_scaler_name.strip().lower()

        candidates = ["custom_robust", "minmax", "median", "mean"]

        # Remove the chosen scaler from the candidates
        if chosen in candidates:
            candidates = [c for c in candidates if c != chosen]
        if not candidates:
            return None

        pick = str(self.rng.choice(candidates))
        if pick == "custom_robust":
            return RobustScaler()
        if pick == "minmax":
            return MinMaxScaler()
        if pick == "median":
            return MedianScaler()
        if pick == "mean":
            return MeanScaler()
        return None

    def _setup_proportions(self, generator_proportions):
        """Setup default or custom generator proportions."""
        default_proportions = {
            "forecast_pfn": 1.0,
            "gp": 1.0,
            "kernel": 1.0,
            "sinewave": 1.0,
            "sawtooth": 1.0,
            "step": 0.1,
            "anomaly": 1.0,
            "spike": 2.0,
            "cauker_univariate": 2.0,
            "cauker_multivariate": 0.00,
            "lmc": 0.00,  # multivariate
            "ou_process": 1.0,
            "audio_financial_volatility": 0.1,
            "audio_multi_scale_fractal": 0.1,
            "audio_network_topology": 0.5,
            "audio_stochastic_rhythm": 1.0,
            "augmented_per_sample_2048": 3.0,
            "augmented_temp_batch_2048": 3.0,
        }
        self.generator_proportions = generator_proportions or default_proportions

        # Normalize proportions
        total = sum(self.generator_proportions.values())
        if total <= 0:
            raise ValueError("Total generator proportions must be positive")
        self.generator_proportions = {k: v / total for k, v in self.generator_proportions.items()}

    def _initialize_datasets(self) -> dict[str, CyclicalBatchDataset]:
        """Initialize CyclicalBatchDataset for each generator with proportion > 0."""
        datasets = {}

        for generator_name, proportion in self.generator_proportions.items():
            # Only initialize datasets for generators with positive proportion
            if proportion <= 0:
                logger.info(f"Skipping {generator_name} (proportion = {proportion})")
                continue

            batches_dir = f"{self.base_data_dir}/{generator_name}"

            try:
                dataset = CyclicalBatchDataset(
                    batches_dir=batches_dir,
                    generator_type=generator_name,
                    device=None,
                    prefetch_next=True,
                    prefetch_threshold=32,
                    rank=self.rank,
                    world_size=self.world_size,
                )
                datasets[generator_name] = dataset
                logger.info(f"Loaded dataset for {generator_name} (proportion = {proportion})")

            except Exception as e:
                logger.warning(f"Failed to load dataset for {generator_name}: {e}")
                continue

        if not datasets:
            raise ValueError(f"No valid datasets found in {self.base_data_dir} or all generators have proportion <= 0")

        return datasets

    def _convert_sample_to_tensors(
        self, sample: dict, future_length: int | None = None
    ) -> tuple[torch.Tensor, np.datetime64, Frequency]:
        """
        Convert a sample dict to tensors and metadata.

        Args:
            sample: Sample dict from CyclicalBatchDataset
            future_length: Desired future length (if None, use default split)

        Returns:
            Tuple of (history_values, future_values, start, frequency)
        """
        # Handle both old and new data formats
        num_channels = sample.get("num_channels", 1)
        values_data = sample["values"]
        generator_type = sample.get("generator_type", "unknown")

        if num_channels == 1:
            # Univariate data
            if isinstance(values_data[0], list):
                # New format: [[channel_values]]
                values = torch.tensor(values_data[0], dtype=torch.float32)
                logger.debug(f"{generator_type}: Using new univariate format, shape: {values.shape}")
            else:
                # Old format: [values]
                values = torch.tensor(values_data, dtype=torch.float32)
            values = values.unsqueeze(0).unsqueeze(-1)  # Shape: [1, seq_len, 1]
        else:
            # Multivariate data (LMC) - new format: [[ch1_values], [ch2_values], ...]
            channel_tensors = []
            for channel_values in values_data:
                channel_tensor = torch.tensor(channel_values, dtype=torch.float32)
                channel_tensors.append(channel_tensor)

            # Stack channels: [1, seq_len, num_channels]
            values = torch.stack(channel_tensors, dim=-1).unsqueeze(0)
            logger.debug(f"{generator_type}: Using multivariate format, {num_channels} channels, shape: {values.shape}")

        # Handle frequency conversion
        freq_str = sample["frequency"]
        try:
            frequency = Frequency(freq_str)
        except ValueError:
            # Map common frequency strings to Frequency enum
            freq_mapping = {
                "h": Frequency.H,
                "D": Frequency.D,
                "W": Frequency.W,
                "M": Frequency.M,
                "Q": Frequency.Q,
                "A": Frequency.A,
                "Y": Frequency.A,  # Annual
                "1min": Frequency.T1,
                "5min": Frequency.T5,
                "10min": Frequency.T10,
                "15min": Frequency.T15,
                "30min": Frequency.T30,
                "s": Frequency.S,
            }
            frequency = freq_mapping.get(freq_str, Frequency.H)  # Default to hourly

        # Handle start timestamp
        if isinstance(sample["start"], pd.Timestamp):
            start = sample["start"].to_numpy()
        else:
            start = np.datetime64(sample["start"])

        return values, start, frequency

    def _effective_proportions_for_length(self, total_length_for_batch: int) -> dict[str, float]:
        """
        Build a simple, length-aware proportion map for the current batch.

        Rules:
        - For generators named 'augmented{L}', keep only the one matching the
          chosen length L; zero out others.
        - Keep non-augmented generators as-is.
        - Drop generators that are unavailable (not loaded) or zero-weight.
        - If nothing remains, fall back to 'augmented{L}' if available, else any dataset.
        - Normalize the final map to sum to 1.
        """

        def augmented_length_from_name(name: str) -> int | None:
            if not name.startswith("augmented"):
                return None
            suffix = name[len("augmented") :]
            if not suffix:
                return None
            try:
                return int(suffix)
            except ValueError:
                return None

        # 1) Adjust proportions with the length-aware rule
        adjusted: dict[str, float] = {}
        for name, proportion in self.generator_proportions.items():
            aug_len = augmented_length_from_name(name)
            if aug_len is None:
                adjusted[name] = proportion
            else:
                adjusted[name] = proportion if aug_len == total_length_for_batch else 0.0

        # 2) Keep only available, positive-weight datasets
        adjusted = {name: p for name, p in adjusted.items() if name in self.datasets and p > 0.0}

        # 3) Fallback if empty
        if not adjusted:
            preferred = f"augmented{total_length_for_batch}"
            if preferred in self.datasets:
                adjusted = {preferred: 1.0}
            elif self.datasets:
                # Choose any available dataset deterministically (first key)
                first_key = next(iter(self.datasets.keys()))
                adjusted = {first_key: 1.0}
            else:
                raise ValueError("No datasets available to create batch")

        # 4) Normalize
        total = sum(adjusted.values())
        return {name: p / total for name, p in adjusted.items()}

    def _compute_sample_counts_for_batch(self, proportions: dict[str, float], batch_size: int) -> dict[str, int]:
        """
        Convert a proportion map into integer sample counts that sum to batch_size.

        Strategy: allocate floor(batch_size * p) to each generator in order, and let the
        last generator absorb any remainder to ensure the total matches exactly.
        """
        counts: dict[str, int] = {}
        remaining = batch_size
        names = list(proportions.keys())
        values = list(proportions.values())
        for index, (name, p) in enumerate(zip(names, values, strict=True)):
            if index == len(names) - 1:
                counts[name] = remaining
            else:
                n = int(batch_size * p)
                counts[name] = n
                remaining -= n
        return counts

    def _calculate_generator_samples(self, batch_size: int) -> dict[str, int]:
        """
        Calculate the number of samples each generator should contribute.

        Args:
            batch_size: Total batch size

        Returns:
            Dict mapping generator names to sample counts
        """
        generator_samples = {}
        remaining_samples = batch_size

        generators = list(self.generator_proportions.keys())
        proportions = list(self.generator_proportions.values())

        # Calculate base samples for each generator
        for i, (generator, proportion) in enumerate(zip(generators, proportions, strict=True)):
            if generator not in self.datasets:
                continue

            if i == len(generators) - 1:  # Last generator gets remaining samples
                samples = remaining_samples
            else:
                samples = int(batch_size * proportion)
                remaining_samples -= samples
            generator_samples[generator] = samples

        return generator_samples

    def create_batch(
        self,
        batch_size: int = 128,
        seed: int | None = None,
        future_length: int | None = None,
    ) -> tuple[BatchTimeSeriesContainer, str]:
        """
        Create a batch of the specified size.

        Args:
            batch_size: Size of the batch to create
            seed: Random seed for this batch
            future_length: Fixed future length to use. If None, samples from gift_eval range

        Returns:
            Tuple of (batch_container, generator_info)
        """
        if seed is not None:
            batch_rng = np.random.default_rng(seed)
            random.seed(seed)
        else:
            batch_rng = self.rng

        if self.mixed_batches:
            return self._create_mixed_batch(batch_size, future_length)
        else:
            return self._create_uniform_batch(batch_size, batch_rng, future_length)

    def _create_mixed_batch(
        self, batch_size: int, future_length: int | None = None
    ) -> tuple[BatchTimeSeriesContainer, str]:
        """Create a mixed batch with samples from multiple generators, rejecting NaNs."""

        # Choose total length for this batch; respect length_shortening flag.
        # When disabled, always use the maximum to avoid shortening.
        if self.augmentations.get("length_shortening", False):
            lengths = list(LENGTH_WEIGHTS.keys())
            probs = list(LENGTH_WEIGHTS.values())
            total_length_for_batch = int(self.rng.choice(lengths, p=probs))
        else:
            total_length_for_batch = int(max(LENGTH_CHOICES))

        if future_length is None:
            prediction_length = int(sample_future_length(range="gift_eval", total_length=total_length_for_batch))
        else:
            prediction_length = future_length

        history_length = total_length_for_batch - prediction_length

        # Calculate samples per generator using simple, per-batch length-aware proportions
        effective_props = self._effective_proportions_for_length(total_length_for_batch)
        generator_samples = self._compute_sample_counts_for_batch(effective_props, batch_size)

        all_values = []
        all_starts = []
        all_frequencies = []
        actual_proportions = {}

        # Collect valid samples from each generator using batched fetches to reduce I/O overhead
        for generator_name, num_samples in generator_samples.items():
            if num_samples == 0 or generator_name not in self.datasets:
                continue

            dataset = self.datasets[generator_name]

            # Lists to hold valid samples for the current generator
            generator_values = []
            generator_starts = []
            generator_frequencies = []

            # Loop until we have collected the required number of VALID samples
            max_attempts = 50
            attempts = 0
            while len(generator_values) < num_samples and attempts < max_attempts:
                attempts += 1
                # Fetch a batch larger than needed to reduce round-trips
                need = num_samples - len(generator_values)
                fetch_n = max(need * 2, 8)
                samples = dataset.get_samples(fetch_n)

                for sample in samples:
                    if len(generator_values) >= num_samples:
                        break

                    values, sample_start, sample_freq = self._convert_sample_to_tensors(sample, future_length)

                    # Skip if NaNs exist (we inject NaNs later in history only)
                    if torch.isnan(values).any():
                        continue

                    # Resize to target batch length when longer
                    if total_length_for_batch < values.shape[1]:
                        strategy = self.rng.choice(["cut", "subsample"])  # 50/50
                        if strategy == "cut":
                            max_start_idx = values.shape[1] - total_length_for_batch
                            start_idx = int(self.rng.integers(0, max_start_idx + 1))
                            values = values[:, start_idx : start_idx + total_length_for_batch, :]
                        else:
                            indices = np.linspace(
                                0,
                                values.shape[1] - 1,
                                total_length_for_batch,
                                dtype=int,
                            )
                            values = values[:, indices, :]

                    # Optionally apply scaler augmentation according to configuration
                    if self._should_apply_scaler_augmentation():
                        scaler = self._choose_random_scaler()
                        if scaler is not None:
                            values = scaler.scale(values, scaler.compute_statistics(values))

                    generator_values.append(values)
                    generator_starts.append(sample_start)
                    generator_frequencies.append(sample_freq)

            if len(generator_values) < num_samples:
                logger.warning(
                    f"Generator {generator_name}: collected {len(generator_values)}/"
                    f"{num_samples} after {attempts} attempts"
                )

            # Add the collected valid samples to the main batch lists
            if generator_values:
                all_values.extend(generator_values)
                all_starts.extend(generator_starts)
                all_frequencies.extend(generator_frequencies)
                actual_proportions[generator_name] = len(generator_values)

        if not all_values:
            raise RuntimeError("No valid samples could be collected from any generator.")

        combined_values = torch.cat(all_values, dim=0)
        # Split into history and future
        combined_history = combined_values[:, :history_length, :]
        combined_future = combined_values[:, history_length : history_length + prediction_length, :]

        if self.nan_augmenter is not None:
            combined_history = self.nan_augmenter.transform(combined_history)

        # Create container
        container = BatchTimeSeriesContainer(
            history_values=combined_history,
            future_values=combined_future,
            start=all_starts,
            frequency=all_frequencies,
        )

        return container, "MixedBatch"

    def _create_uniform_batch(
        self,
        batch_size: int,
        batch_rng: np.random.Generator,
        future_length: int | None = None,
    ) -> tuple[BatchTimeSeriesContainer, str]:
        """Create a uniform batch with samples from a single generator."""

        # Select generator based on proportions
        generators = list(self.datasets.keys())
        proportions = [self.generator_proportions[gen] for gen in generators]
        selected_generator = batch_rng.choice(generators, p=proportions)

        # Sample future length
        if future_length is None:
            future_length = sample_future_length(range="gift_eval")

        # Get samples from selected generator
        dataset = self.datasets[selected_generator]
        samples = dataset.get_samples(batch_size)

        all_history_values = []
        all_future_values = []
        all_starts = []
        all_frequencies = []

        for sample in samples:
            values, sample_start, sample_freq = self._convert_sample_to_tensors(sample, future_length)

            total_length = values.shape[1]
            history_length = max(1, total_length - future_length)

            # Optionally apply scaler augmentation according to configuration
            if self._should_apply_scaler_augmentation():
                scaler = self._choose_random_scaler()
                if scaler is not None:
                    values = scaler.scale(values, scaler.compute_statistics(values))

            # Reshape to [1, seq_len, 1] for single sample
            hist_vals = values[:, :history_length, :]
            fut_vals = values[:, history_length : history_length + future_length, :]

            all_history_values.append(hist_vals)
            all_future_values.append(fut_vals)
            all_starts.append(sample_start)
            all_frequencies.append(sample_freq)

        # Combine samples
        combined_history = torch.cat(all_history_values, dim=0)
        combined_future = torch.cat(all_future_values, dim=0)

        # Create container
        container = BatchTimeSeriesContainer(
            history_values=combined_history,
            future_values=combined_future,
            start=all_starts,
            frequency=all_frequencies,
        )

        return container, selected_generator

    def get_dataset_info(self) -> dict[str, dict]:
        """Get information about all datasets."""
        info = {}
        for name, dataset in self.datasets.items():
            info[name] = dataset.get_info()
        return info

    def get_generator_info(self) -> dict[str, any]:
        """Get information about the composer configuration."""
        return {
            "mixed_batches": self.mixed_batches,
            "generator_proportions": self.generator_proportions,
            "active_generators": list(self.datasets.keys()),
            "total_generators": len(self.datasets),
            "augmentations": self.augmentations,
            "augmentation_probabilities": self.augmentation_probabilities,
            "nan_augmenter_enabled": self.nan_augmenter is not None,
        }


class ComposedDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset wrapper around BatchComposer for training pipeline integration.
    """

    def __init__(
        self,
        batch_composer: BatchComposer,
        num_batches_per_epoch: int = 100,
        batch_size: int = 128,
    ):
        """
        Initialize the dataset.

        Args:
            batch_composer: The BatchComposer instance
            num_batches_per_epoch: Number of batches to generate per epoch
            batch_size: Size of each batch
        """
        self.batch_composer = batch_composer
        self.num_batches_per_epoch = num_batches_per_epoch
        self.batch_size = batch_size

    def __len__(self) -> int:
        return self.num_batches_per_epoch

    def __getitem__(self, idx: int) -> BatchTimeSeriesContainer:
        """
        Get a batch by index.

        Args:
            idx: Batch index (used as seed for reproducibility)

        Returns:
            BatchTimeSeriesContainer
        """
        # Use index as seed for reproducible batches
        batch, _ = self.batch_composer.create_batch(
            batch_size=self.batch_size, seed=self.batch_composer.global_seed + idx
        )
        return batch
