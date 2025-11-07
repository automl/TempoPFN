import logging
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed
from torch.quasirandom import SobolEngine

from src.gift_eval.data import Dataset

logger = logging.getLogger(__name__)


def find_consecutive_nan_lengths(series: np.ndarray) -> list[int]:
    """Finds the lengths of all consecutive NaN blocks in a 1D array."""
    if series.ndim > 1:
        # For multivariate series, flatten to treat it as one long sequence
        series = series.flatten()

    is_nan = np.isnan(series)
    padded_is_nan = np.concatenate(([False], is_nan, [False]))
    diffs = np.diff(padded_is_nan.astype(int))

    start_indices = np.where(diffs == 1)[0]
    end_indices = np.where(diffs == -1)[0]

    return (end_indices - start_indices).tolist()


def analyze_datasets_for_augmentation(gift_eval_path_str: str) -> dict:
    """
    Analyzes all datasets to derive statistics needed for NaN augmentation.
    This version collects the full distribution of NaN ratios.
    """
    logger.info("--- Starting Dataset Analysis for Augmentation (Full Distribution) ---")
    path = Path(gift_eval_path_str)
    if not path.exists():
        raise FileNotFoundError(
            f"Provided raw data path for augmentation analysis does not exist: {gift_eval_path_str}"
        )

    dataset_names = []
    for dataset_dir in path.iterdir():
        if dataset_dir.name.startswith(".") or not dataset_dir.is_dir():
            continue
        freq_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
        if freq_dirs:
            for freq_dir in freq_dirs:
                dataset_names.append(f"{dataset_dir.name}/{freq_dir.name}")
        else:
            dataset_names.append(dataset_dir.name)

    total_series_count = 0
    series_with_nans_count = 0
    nan_ratio_distribution = []
    all_consecutive_nan_lengths = Counter()

    for ds_name in sorted(dataset_names):
        try:
            ds = Dataset(name=ds_name, term="short", to_univariate=False)
            for series_data in ds.training_dataset:
                total_series_count += 1
                target = np.atleast_1d(series_data["target"])
                num_nans = np.isnan(target).sum()

                if num_nans > 0:
                    series_with_nans_count += 1
                    nan_ratio = num_nans / target.size
                    nan_ratio_distribution.append(float(nan_ratio))

                    nan_lengths = find_consecutive_nan_lengths(target)
                    all_consecutive_nan_lengths.update(nan_lengths)
        except Exception as e:
            logger.warning(f"Could not process {ds_name} for augmentation analysis: {e}")

    if total_series_count == 0:
        raise ValueError("No series were found during augmentation analysis. Check dataset path.")

    p_series_has_nan = series_with_nans_count / total_series_count if total_series_count > 0 else 0

    logger.info("--- Augmentation Analysis Complete ---")
    # Print summary statistics
    logger.info(f"Total series analyzed: {total_series_count}")
    logger.info(f"Series with NaNs: {series_with_nans_count} ({p_series_has_nan:.4f})")
    logger.info(f"NaN ratio distribution: {Counter(nan_ratio_distribution)}")
    logger.info(f"Consecutive NaN lengths distribution: {all_consecutive_nan_lengths}")
    logger.info("--- End of Dataset Analysis for Augmentation ---")
    return {
        "p_series_has_nan": p_series_has_nan,
        "nan_ratio_distribution": nan_ratio_distribution,
        "nan_length_distribution": all_consecutive_nan_lengths,
    }


class NanAugmenter:
    """
    Applies realistic NaN augmentation by generating and caching NaN patterns on-demand
    during the first transform call for a given data shape.
    """

    def __init__(
        self,
        p_series_has_nan: float,
        nan_ratio_distribution: list[float],
        nan_length_distribution: Counter,
        num_patterns: int = 100000,
        n_jobs: int = -1,
        nan_patterns_path: str | None = None,
    ):
        """
        Initializes the augmenter. NaN patterns are not generated at this stage.

        Args:
            p_series_has_nan (float): Probability that a series in a batch will be augmented.
            nan_ratio_distribution (List[float]): A list of NaN ratios observed in the dataset.
            nan_length_distribution (Counter): A Counter of consecutive NaN block lengths.
            num_patterns (int): The number of unique NaN patterns to generate per data shape.
            n_jobs (int): The number of CPU cores to use for parallel pattern generation (-1 for all cores).
        """
        self.p_series_has_nan = p_series_has_nan
        self.nan_ratio_distribution = nan_ratio_distribution
        self.num_patterns = num_patterns
        self.n_jobs = n_jobs
        self.max_length = 2048
        self.nan_patterns_path = nan_patterns_path
        # Cache to store patterns: Dict[shape_tuple -> pattern_tensor]
        self.pattern_cache: dict[tuple[int, ...], torch.BoolTensor] = {}

        if not nan_length_distribution or sum(nan_length_distribution.values()) == 0:
            self._has_block_distribution = False
            logger.warning("NaN length distribution is empty. Augmentation disabled.")
        else:
            self._has_block_distribution = True
            total_blocks = sum(nan_length_distribution.values())
            self.dist_lengths = [int(i) for i in nan_length_distribution.keys()]
            self.dist_probs = [count / total_blocks for count in nan_length_distribution.values()]

        if not self.nan_ratio_distribution:
            logger.warning("NaN ratio distribution is empty. Augmentation disabled.")

        # Try to load existing patterns from disk
        self._load_existing_patterns()

    def _load_existing_patterns(self):
        """Load existing NaN patterns from disk if they exist."""
        # Determine where to look for patterns
        explicit_path: Path | None = (
            Path(self.nan_patterns_path).resolve() if self.nan_patterns_path is not None else None
        )

        candidate_files: list[Path] = []
        if explicit_path is not None:
            # If the explicit path exists, use it directly
            if explicit_path.is_file():
                candidate_files.append(explicit_path)
            # Also search the directory of the explicit path for matching files
            explicit_dir = explicit_path.parent
            explicit_dir.mkdir(exist_ok=True, parents=True)
            candidate_files.extend(list(explicit_dir.glob(f"nan_patterns_{self.max_length}_*.pt")))
        else:
            # Default to the ./data directory
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            candidate_files.extend(list(data_dir.glob(f"nan_patterns_{self.max_length}_*.pt")))

        # De-duplicate candidate files while preserving order
        seen: set[str] = set()
        unique_candidates: list[Path] = []
        for f in candidate_files:
            key = str(f.resolve())
            if key not in seen:
                seen.add(key)
                unique_candidates.append(f)

        for pattern_file in unique_candidates:
            try:
                # Extract num_channels from filename
                filename = pattern_file.stem
                parts = filename.split("_")
                if len(parts) >= 4:
                    num_channels = int(parts[-1])

                    # Load patterns
                    patterns = torch.load(pattern_file, map_location="cpu")
                    cache_key = (self.max_length, num_channels)
                    self.pattern_cache[cache_key] = patterns

                    logger.info(f"Loaded {patterns.shape[0]} patterns for shape {cache_key} from {pattern_file}")
            except (ValueError, RuntimeError, FileNotFoundError) as e:
                logger.warning(f"Failed to load patterns from {pattern_file}: {e}")

    def _get_pattern_file_path(self, num_channels: int) -> Path:
        """Resolve the target file path for storing/loading patterns for a given channel count."""
        # If user provided a file path, use its directory as the base directory
        if self.nan_patterns_path is not None:
            base_dir = Path(self.nan_patterns_path).resolve().parent
            base_dir.mkdir(exist_ok=True, parents=True)
        else:
            base_dir = Path("data").resolve()
            base_dir.mkdir(exist_ok=True, parents=True)

        return base_dir / f"nan_patterns_{self.max_length}_{num_channels}.pt"

    def _generate_nan_mask(self, series_shape: tuple[int, ...]) -> np.ndarray:
        """Generates a single boolean NaN mask for a given series shape."""
        series_size = int(np.prod(series_shape))
        sampled_ratio = np.random.choice(self.nan_ratio_distribution)
        n_nans_to_add = int(round(series_size * sampled_ratio))

        if n_nans_to_add == 0:
            return np.zeros(series_shape, dtype=bool)

        mask_flat = np.zeros(series_size, dtype=bool)
        nans_added = 0
        max_attempts = n_nans_to_add * 2
        attempts = 0
        while nans_added < n_nans_to_add and attempts < max_attempts:
            attempts += 1
            block_length = np.random.choice(self.dist_lengths, p=self.dist_probs)

            if nans_added + block_length > n_nans_to_add:
                block_length = n_nans_to_add - nans_added
            if block_length <= 0:
                break

            nan_counts_in_window = np.convolve(mask_flat, np.ones(block_length), mode="valid")
            valid_starts = np.where(nan_counts_in_window == 0)[0]

            if valid_starts.size == 0:
                continue

            start_pos = np.random.choice(valid_starts)
            mask_flat[start_pos : start_pos + block_length] = True
            nans_added += block_length

        return mask_flat.reshape(series_shape)

    def _pregenerate_patterns(self, series_shape: tuple[int, ...]) -> torch.BoolTensor:
        """Uses joblib to parallelize the generation of NaN masks for a given shape."""
        if not self._has_block_distribution or not self.nan_ratio_distribution:
            return torch.empty(0, *series_shape, dtype=torch.bool)

        logger.info(f"Generating {self.num_patterns} NaN patterns for shape {series_shape}...")

        with Parallel(n_jobs=self.n_jobs, backend="loky") as parallel:
            masks_list = parallel(delayed(self._generate_nan_mask)(series_shape) for _ in range(self.num_patterns))

        logger.info(f"Pattern generation complete for shape {series_shape}.")
        return torch.from_numpy(np.stack(masks_list)).bool()

    def transform(self, time_series_batch: torch.Tensor) -> torch.Tensor:
        """
        Applies NaN patterns to a batch, generating them on-demand if the shape is new.
        """
        if self.p_series_has_nan == 0:
            return time_series_batch

        history_length, num_channels = time_series_batch.shape[1:]
        assert history_length <= self.max_length, (
            f"History length {history_length} exceeds maximum allowed {self.max_length}."
        )

        # 1. Check cache and generate patterns if the shape is new
        if (
            self.max_length,
            num_channels,
        ) not in self.pattern_cache:
            # Try loading from a resolved file path if available
            target_file = self._get_pattern_file_path(num_channels)
            if target_file.exists():
                try:
                    patterns = torch.load(target_file, map_location="cpu")
                    self.pattern_cache[(self.max_length, num_channels)] = patterns
                    logger.info(f"Loaded NaN patterns from {target_file} for shape {(self.max_length, num_channels)}")
                except (RuntimeError, FileNotFoundError):
                    # Fall back to generating if loading fails
                    patterns = self._pregenerate_patterns((self.max_length, num_channels))
                    torch.save(patterns, target_file)
                    self.pattern_cache[(self.max_length, num_channels)] = patterns
                    logger.info(f"Generated and saved {patterns.shape[0]} NaN patterns to {target_file}")
            else:
                patterns = self._pregenerate_patterns((self.max_length, num_channels))
                torch.save(patterns, target_file)
                self.pattern_cache[(self.max_length, num_channels)] = patterns
                logger.info(f"Generated and saved {patterns.shape[0]} NaN patterns to {target_file}")
        patterns = self.pattern_cache[(self.max_length, num_channels)][:, :history_length, :]

        # Early exit if patterns are empty (e.g., generation failed or was disabled)
        if patterns.numel() == 0:
            return time_series_batch

        batch_size = time_series_batch.shape[0]
        device = time_series_batch.device

        # 2. Vectorized decision on which series to augment
        augment_mask = torch.rand(batch_size, device=device) < self.p_series_has_nan
        indices_to_augment = torch.where(augment_mask)[0]
        num_to_augment = indices_to_augment.numel()

        if num_to_augment == 0:
            return time_series_batch

        # 3. Randomly sample patterns for each series being augmented
        pattern_indices = torch.randint(0, patterns.shape[0], (num_to_augment,), device=device)
        # 4. Select patterns and apply them in a single vectorized operation
        selected_patterns = patterns[pattern_indices].to(device)

        time_series_batch[indices_to_augment] = time_series_batch[indices_to_augment].masked_fill(
            selected_patterns, float("nan")
        )

        return time_series_batch


class CensorAugmenter:
    """
    Applies censor augmentation by clipping values from above, below, or both.
    """

    def __init__(self):
        """Initializes the CensorAugmenter."""
        pass

    def transform(self, time_series_batch: torch.Tensor) -> torch.Tensor:
        """
        Applies a vectorized censor augmentation to a batch of time series.
        """
        batch_size, seq_len, num_channels = time_series_batch.shape
        assert num_channels == 1
        time_series_batch = time_series_batch.squeeze(-1)
        with torch.no_grad():
            batch_size, seq_len = time_series_batch.shape
            device = time_series_batch.device

            # Step 1: Choose an op mode for each series
            op_mode = torch.randint(0, 3, (batch_size, 1), device=device)

            # Step 2: Calculate potential thresholds for all series
            q1 = torch.rand(batch_size, device=device)
            q2 = torch.rand(batch_size, device=device)
            q_low = torch.minimum(q1, q2)
            q_high = torch.maximum(q1, q2)

            sorted_series = torch.sort(time_series_batch, dim=1).values
            indices_low = (q_low * (seq_len - 1)).long()
            indices_high = (q_high * (seq_len - 1)).long()

            c_low = torch.gather(sorted_series, 1, indices_low.unsqueeze(1))
            c_high = torch.gather(sorted_series, 1, indices_high.unsqueeze(1))

            # Step 3: Compute results for all possible clipping operations
            clip_above = torch.minimum(time_series_batch, c_high)
            clip_below = torch.maximum(time_series_batch, c_low)

            # Step 4: Select the final result based on the op_mode
            result = torch.where(
                op_mode == 1,
                clip_above,
                torch.where(op_mode == 2, clip_below, time_series_batch),
            )
            augmented_batch = torch.where(
                op_mode == 0,
                time_series_batch,
                result,
            )

        return augmented_batch.unsqueeze(-1)


class QuantizationAugmenter:
    """
    Applies non-equidistant quantization using a Sobol sequence to generate
    uniformly distributed levels. This implementation is fully vectorized.
    """

    def __init__(
        self,
        p_quantize: float,
        level_range: tuple[int, int],
        seed: int | None = None,
    ):
        """
        Initializes the augmenter.

        Args:
            p_quantize (float): Probability of applying quantization to a series.
            level_range (Tuple[int, int]): Inclusive range [min, max] to sample the
                                           number of quantization levels from.
            seed (Optional[int]): Seed for the Sobol sequence generator for reproducibility.
        """
        assert 0.0 <= p_quantize <= 1.0, "Probability must be between 0 and 1."
        assert level_range[0] >= 2, "Minimum number of levels must be at least 2."
        assert level_range[0] <= level_range[1], "Min levels cannot be greater than max."

        self.p_quantize = p_quantize
        self.level_range = level_range

        # Initialize a SobolEngine. The dimension is the max number of random
        # levels we might need to generate for a single series.
        max_intermediate_levels = self.level_range[1] - 2
        if max_intermediate_levels > 0:
            # SobolEngine must be created on CPU
            self.sobol_engine = SobolEngine(dimension=max_intermediate_levels, scramble=True, seed=seed)
        else:
            self.sobol_engine = None

    def transform(self, time_series_batch: torch.Tensor) -> torch.Tensor:
        """
        Applies augmentation in a fully vectorized way on the batch's device.
        Handles input shape (batch, length, 1).
        """
        # Handle input shape (batch, length, 1)
        if time_series_batch.dim() == 3 and time_series_batch.shape[2] == 1:
            is_3d = True
            time_series_squeezed = time_series_batch.squeeze(-1)
        else:
            is_3d = False
            time_series_squeezed = time_series_batch

        if self.p_quantize == 0 or self.sobol_engine is None:
            return time_series_batch

        n_series, _ = time_series_squeezed.shape
        device = time_series_squeezed.device

        # 1. Decide which series to augment
        augment_mask = torch.rand(n_series, device=device) < self.p_quantize
        n_augment = torch.sum(augment_mask)
        if n_augment == 0:
            return time_series_batch

        series_to_augment = time_series_squeezed[augment_mask]

        # 2. Determine a variable n_levels for EACH series
        min_l, max_l = self.level_range
        n_levels_per_series = torch.randint(min_l, max_l + 1, size=(n_augment,), device=device)
        max_levels_in_batch = n_levels_per_series.max().item()

        # 3. Find min/max for each series
        min_vals = torch.amin(series_to_augment, dim=1, keepdim=True)
        max_vals = torch.amax(series_to_augment, dim=1, keepdim=True)
        value_range = max_vals - min_vals
        is_flat = value_range == 0

        # 4. Generate quasi-random levels using the Sobol sequence
        num_intermediate_levels = max_levels_in_batch - 2
        if num_intermediate_levels > 0:
            # Draw points from the Sobol engine (on CPU) and move to target device
            sobol_points = self.sobol_engine.draw(n_augment).to(device)
            # We only need the first `num_intermediate_levels` dimensions
            quasi_rand_points = sobol_points[:, :num_intermediate_levels]
        else:
            # Handle case where max_levels_in_batch is 2 (no intermediate points needed)
            quasi_rand_points = torch.empty(n_augment, 0, device=device)

        scaled_quasi_rand_levels = min_vals + value_range * quasi_rand_points
        level_values = torch.cat([min_vals, max_vals, scaled_quasi_rand_levels], dim=1)
        level_values, _ = torch.sort(level_values, dim=1)

        # 5. Find the closest level using a mask to ignore padded values
        series_expanded = series_to_augment.unsqueeze(2)
        levels_expanded = level_values.unsqueeze(1)
        diff = torch.abs(series_expanded - levels_expanded)

        arange_mask = torch.arange(max_levels_in_batch, device=device).unsqueeze(0)
        valid_levels_mask = arange_mask < n_levels_per_series.unsqueeze(1)
        masked_diff = torch.where(valid_levels_mask.unsqueeze(1), diff, float("inf"))
        closest_level_indices = torch.argmin(masked_diff, dim=2)

        # 6. Gather the results from the original level values
        quantized_subset = torch.gather(level_values, 1, closest_level_indices)

        # 7. For flat series, revert to their original values
        final_subset = torch.where(is_flat, series_to_augment, quantized_subset)

        # 8. Place augmented data back into a copy of the original batch
        augmented_batch_squeezed = time_series_squeezed.clone()
        augmented_batch_squeezed[augment_mask] = final_subset

        # Restore original shape before returning
        if is_3d:
            return augmented_batch_squeezed.unsqueeze(-1)
        else:
            return augmented_batch_squeezed


class MixUpAugmenter:
    """
    Applies mixup augmentation by creating a weighted average of multiple time series.

    This version includes an option for time-dependent mixup using Simplex Path
    Interpolation, creating a smooth transition between different mixing weights.
    """

    def __init__(
        self,
        max_n_series_to_combine: int = 10,
        p_combine: float = 0.4,
        p_time_dependent: float = 0.5,
        randomize_k_per_series: bool = True,
        dirichlet_alpha_range: tuple[float, float] = (0.1, 5.0),
    ):
        """
        Initializes the augmenter.

        Args:
            max_n_series_to_combine (int): The maximum number of series to combine.
                The actual number k will be sampled from [2, max].
            p_combine (float): The probability of replacing a series with a combination.
            p_time_dependent (float): The probability of using the time-dependent
                simplex path method for a given mixup operation. Defaults to 0.5.
            randomize_k_per_series (bool): If True, each augmented series will be a
                combination of a different number of series (k).
                If False, one k is chosen for the whole batch.
            dirichlet_alpha_range (Tuple[float, float]): The [min, max] range to sample the
                Dirichlet 'alpha' from. A smaller alpha (e.g., 0.2) creates mixes
                dominated by one series. A larger alpha (e.g., 5.0) creates
                more uniform weights.
        """
        assert max_n_series_to_combine >= 2, "Must combine at least 2 series."
        assert 0.0 <= p_combine <= 1.0, "p_combine must be between 0 and 1."
        assert 0.0 <= p_time_dependent <= 1.0, "p_time_dependent must be between 0 and 1."
        assert dirichlet_alpha_range[0] > 0 and dirichlet_alpha_range[0] <= dirichlet_alpha_range[1]
        self.max_k = max_n_series_to_combine
        self.p_combine = p_combine
        self.p_time_dependent = p_time_dependent
        self.randomize_k = randomize_k_per_series
        self.alpha_range = dirichlet_alpha_range

    def _sample_alpha(self) -> float:
        log_alpha_min = math.log10(self.alpha_range[0])
        log_alpha_max = math.log10(self.alpha_range[1])
        log_alpha = log_alpha_min + np.random.rand() * (log_alpha_max - log_alpha_min)
        return float(10**log_alpha)

    def _sample_k(self) -> int:
        return int(torch.randint(2, self.max_k + 1, (1,)).item())

    def _static_mix(
        self,
        source_series: torch.Tensor,
        alpha: float,
        return_weights: bool = False,
    ):
        """Mixes k source series using a single, static set of Dirichlet weights."""
        k = int(source_series.shape[0])
        device = source_series.device
        concentration = torch.full((k,), float(alpha), device=device)
        weights = torch.distributions.Dirichlet(concentration).sample()
        weights_view = weights.view(k, 1, 1)
        mixed_series = (source_series * weights_view).sum(dim=0, keepdim=True)
        if return_weights:
            return mixed_series, weights
        return mixed_series

    def _simplex_path_mix(
        self,
        source_series: torch.Tensor,
        alpha: float,
        return_weights: bool = False,
    ):
        """Mixes k series using time-varying weights interpolated along a simplex path."""
        k, length, _ = source_series.shape
        device = source_series.device

        # 1. Sample two endpoint weight vectors from the Dirichlet distribution
        concentration = torch.full((k,), float(alpha), device=device)
        dirichlet_dist = torch.distributions.Dirichlet(concentration)
        w_start = dirichlet_dist.sample()
        w_end = dirichlet_dist.sample()

        # 2. Create a linear ramp from 0 to 1
        alpha_ramp = torch.linspace(0, 1, length, device=device)

        # 3. Interpolate between the endpoint weights over time
        # Reshape for broadcasting: w vectors become [k, 1], ramp becomes [1, length]
        time_varying_weights = w_start.unsqueeze(1) * (1 - alpha_ramp.unsqueeze(0)) + w_end.unsqueeze(
            1
        ) * alpha_ramp.unsqueeze(0)
        # The result `time_varying_weights` has shape [k, length]

        # 4. Apply the time-varying weights
        weights_view = time_varying_weights.unsqueeze(-1)  # Shape: [k, length, 1]
        mixed_series = (source_series * weights_view).sum(dim=0, keepdim=True)

        if return_weights:
            return mixed_series, time_varying_weights
        return mixed_series

    def transform(self, time_series_batch: torch.Tensor, return_debug_info: bool = False):
        """
        Applies the mixup augmentation, randomly choosing between static and
        time-dependent mixing methods.
        """
        with torch.no_grad():
            if self.p_combine == 0:
                return (time_series_batch, {}) if return_debug_info else time_series_batch

            batch_size, _, _ = time_series_batch.shape
            device = time_series_batch.device

            if batch_size <= self.max_k:
                return (time_series_batch, {}) if return_debug_info else time_series_batch

            # 1. Decide which series to replace
            augment_mask = torch.rand(batch_size, device=device) < self.p_combine
            indices_to_replace = torch.where(augment_mask)[0]
            n_augment = indices_to_replace.numel()

            if n_augment == 0:
                return (time_series_batch, {}) if return_debug_info else time_series_batch

            # 2. Determine k for each series to augment
            if self.randomize_k:
                k_values = torch.randint(2, self.max_k + 1, (n_augment,), device=device)
            else:
                k = self._sample_k()
                k_values = torch.full((n_augment,), k, device=device)

            # 3. Augment series one by one
            new_series_list = []
            all_batch_indices = torch.arange(batch_size, device=device)
            debug_info = {}

            for i, target_idx in enumerate(indices_to_replace):
                current_k = k_values[i].item()

                # Sample source indices
                candidate_mask = all_batch_indices != target_idx
                candidates = all_batch_indices[candidate_mask]
                perm = torch.randperm(candidates.shape[0], device=device)
                source_indices = candidates[perm[:current_k]]
                source_series = time_series_batch[source_indices]

                alpha = self._sample_alpha()
                mix_type = "static"

                # Randomly choose between static and time-dependent mixup
                if torch.rand(1).item() < self.p_time_dependent:
                    mixed_series, weights = self._simplex_path_mix(source_series, alpha=alpha, return_weights=True)
                    mix_type = "simplex"
                else:
                    mixed_series, weights = self._static_mix(source_series, alpha=alpha, return_weights=True)

                new_series_list.append(mixed_series)

                if return_debug_info:
                    debug_info[target_idx.item()] = {
                        "source_indices": source_indices.cpu().numpy(),
                        "weights": weights.cpu().numpy(),
                        "alpha": alpha,
                        "k": current_k,
                        "mix_type": mix_type,
                    }

            # 4. Place augmented series back into a clone of the original batch
            augmented_batch = time_series_batch.clone()
            if new_series_list:
                new_series_tensor = torch.cat(new_series_list, dim=0)
                augmented_batch[indices_to_replace] = new_series_tensor

            if return_debug_info:
                return augmented_batch.detach(), debug_info
            return augmented_batch.detach()


class TimeFlipAugmenter:
    """
    Applies time-reversal augmentation to a random subset of time series in a batch.
    """

    def __init__(self, p_flip: float = 0.5):
        """
        Initializes the TimeFlipAugmenter.

        Args:
            p_flip (float): The probability of flipping a single time series in the batch.
                            Defaults to 0.5.
        """
        assert 0.0 <= p_flip <= 1.0, "Probability must be between 0 and 1."
        self.p_flip = p_flip

    def transform(self, time_series_batch: torch.Tensor) -> torch.Tensor:
        """
        Applies time-reversal augmentation to a batch of time series.

        Args:
            time_series_batch (torch.Tensor): The input batch of time series with
                                              shape (batch_size, seq_len, num_channels).

        Returns:
            torch.Tensor: The batch with some series potentially flipped.
        """
        with torch.no_grad():
            if self.p_flip == 0:
                return time_series_batch

            batch_size = time_series_batch.shape[0]
            device = time_series_batch.device

            # 1. Decide which series in the batch to flip
            flip_mask = torch.rand(batch_size, device=device) < self.p_flip
            indices_to_flip = torch.where(flip_mask)[0]

            if indices_to_flip.numel() == 0:
                return time_series_batch

            # 2. Select the series to be flipped
            series_to_flip = time_series_batch[indices_to_flip]

            # 3. Flip them along the time dimension (dim=1)
            flipped_series = torch.flip(series_to_flip, dims=[1])

            # 4. Create a copy of the batch and place the flipped series into it
            augmented_batch = time_series_batch.clone()
            augmented_batch[indices_to_flip] = flipped_series

            return augmented_batch


class YFlipAugmenter:
    """
    Applies y-reversal augmentation to a random subset of time series in a batch.
    """

    def __init__(self, p_flip: float = 0.5):
        """
        Initializes the TimeFlipAugmenter.

        Args:
            p_flip (float): The probability of flipping a single time series in the batch.
                            Defaults to 0.5.
        """
        assert 0.0 <= p_flip <= 1.0, "Probability must be between 0 and 1."
        self.p_flip = p_flip

    def transform(self, time_series_batch: torch.Tensor) -> torch.Tensor:
        """
        Applies time-reversal augmentation to a batch of time series.

        Args:
            time_series_batch (torch.Tensor): The input batch of time series with
                                              shape (batch_size, seq_len, num_channels).

        Returns:
            torch.Tensor: The batch with some series potentially flipped.
        """
        with torch.no_grad():
            if self.p_flip == 0:
                return time_series_batch

            batch_size = time_series_batch.shape[0]
            device = time_series_batch.device

            # 1. Decide which series in the batch to flip
            flip_mask = torch.rand(batch_size, device=device) < self.p_flip
            indices_to_flip = torch.where(flip_mask)[0]

            if indices_to_flip.numel() == 0:
                return time_series_batch

            # 2. Select the series to be flipped
            series_to_flip = time_series_batch[indices_to_flip]

            # 3. Flip them along the time dimension (dim=1)
            flipped_series = -series_to_flip

            # 4. Create a copy of the batch and place the flipped series into it
            augmented_batch = time_series_batch.clone()
            augmented_batch[indices_to_flip] = flipped_series

            return augmented_batch


class DifferentialAugmenter:
    """
    Applies calculus-inspired augmentations. This version includes up to the
    fourth derivative and uses nn.Conv1d with built-in 'reflect' padding for
    cleaner and more efficient convolutions.

    The Gaussian kernel size and sigma for the initial smoothing are randomly
    sampled at every transform() call from user-defined ranges.
    """

    def __init__(
        self,
        p_transform: float,
        gaussian_kernel_size_range: tuple[int, int] = (5, 51),
        gaussian_sigma_range: tuple[float, float] = (2.0, 20.0),
    ):
        """
        Initializes the augmenter.

        Args:
            p_transform (float): The probability of applying an augmentation to any given
                                 time series in a batch.
            gaussian_kernel_size_range (Tuple[int, int]): The [min, max] inclusive range
                                                           for the Gaussian kernel size.
                                                           Sizes will be forced to be odd.
            gaussian_sigma_range (Tuple[float, float]): The [min, max] inclusive range
                                                        for the Gaussian sigma.
        """
        self.p_transform = p_transform
        self.kernel_size_range = gaussian_kernel_size_range
        self.sigma_range = gaussian_sigma_range

        # Validate ranges
        if not (self.kernel_size_range[0] <= self.kernel_size_range[1] and self.kernel_size_range[0] >= 3):
            raise ValueError("Invalid kernel size range. Ensure min <= max and min >= 3.")
        if not (self.sigma_range[0] <= self.sigma_range[1] and self.sigma_range[0] > 0):
            raise ValueError("Invalid sigma range. Ensure min <= max and min > 0.")

        # Cache for fixed-kernel convolution layers (Sobel, Laplace, etc.)
        self.conv_cache: dict[tuple[int, torch.device], dict[str, nn.Module]] = {}

    def _create_fixed_kernel_layers(self, num_channels: int, device: torch.device) -> dict:
        """
        Creates and configures nn.Conv1d layers for fixed-kernel derivative operations.
        These layers are cached to improve performance.
        """
        sobel_conv = nn.Conv1d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=3,
            padding="same",
            padding_mode="reflect",
            groups=num_channels,
            bias=False,
            device=device,
        )
        laplace_conv = nn.Conv1d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=3,
            padding="same",
            padding_mode="reflect",
            groups=num_channels,
            bias=False,
            device=device,
        )
        d3_conv = nn.Conv1d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=5,
            padding="same",
            padding_mode="reflect",
            groups=num_channels,
            bias=False,
            device=device,
        )
        d4_conv = nn.Conv1d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=5,
            padding="same",
            padding_mode="reflect",
            groups=num_channels,
            bias=False,
            device=device,
        )

        sobel_kernel = (
            torch.tensor([-1, 0, 1], device=device, dtype=torch.float32).view(1, 1, -1).repeat(num_channels, 1, 1)
        )
        laplace_kernel = (
            torch.tensor([1, -2, 1], device=device, dtype=torch.float32).view(1, 1, -1).repeat(num_channels, 1, 1)
        )
        d3_kernel = (
            torch.tensor([-1, 2, 0, -2, 1], device=device, dtype=torch.float32)
            .view(1, 1, -1)
            .repeat(num_channels, 1, 1)
        )
        d4_kernel = (
            torch.tensor([1, -4, 6, -4, 1], device=device, dtype=torch.float32)
            .view(1, 1, -1)
            .repeat(num_channels, 1, 1)
        )

        sobel_conv.weight.data = sobel_kernel
        laplace_conv.weight.data = laplace_kernel
        d3_conv.weight.data = d3_kernel
        d4_conv.weight.data = d4_kernel

        for layer in [sobel_conv, laplace_conv, d3_conv, d4_conv]:
            layer.weight.requires_grad = False

        return {
            "sobel": sobel_conv,
            "laplace": laplace_conv,
            "d3": d3_conv,
            "d4": d4_conv,
        }

    def _create_gaussian_layer(
        self, kernel_size: int, sigma: float, num_channels: int, device: torch.device
    ) -> nn.Module:
        """Creates a single Gaussian convolution layer with the given dynamic parameters."""
        gauss_conv = nn.Conv1d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=kernel_size,
            padding="same",
            padding_mode="reflect",
            groups=num_channels,
            bias=False,
            device=device,
        )
        ax = torch.arange(
            -(kernel_size // 2),
            kernel_size // 2 + 1,
            device=device,
            dtype=torch.float32,
        )
        gauss_kernel = torch.exp(-0.5 * (ax / sigma) ** 2)
        gauss_kernel /= gauss_kernel.sum()
        gauss_kernel = gauss_kernel.view(1, 1, -1).repeat(num_channels, 1, 1)
        gauss_conv.weight.data = gauss_kernel
        gauss_conv.weight.requires_grad = False
        return gauss_conv

    def _rescale_signal(self, processed_signal: torch.Tensor, original_signal: torch.Tensor) -> torch.Tensor:
        """Rescales the processed signal to match the min/max range of the original."""
        original_min = torch.amin(original_signal, dim=2, keepdim=True)
        original_max = torch.amax(original_signal, dim=2, keepdim=True)
        processed_min = torch.amin(processed_signal, dim=2, keepdim=True)
        processed_max = torch.amax(processed_signal, dim=2, keepdim=True)

        original_range = original_max - original_min
        processed_range = processed_max - processed_min
        epsilon = 1e-8
        rescaled_signal = (
            (processed_signal - processed_min) / (processed_range + epsilon)
        ) * original_range + original_min
        return torch.where(original_range < epsilon, original_signal, rescaled_signal)

    def transform(self, time_series_batch: torch.Tensor) -> torch.Tensor:
        """Applies a random augmentation to a subset of the batch."""
        with torch.no_grad():
            if self.p_transform == 0:
                return time_series_batch

            batch_size, seq_len, num_channels = time_series_batch.shape
            device = time_series_batch.device

            augment_mask = torch.rand(batch_size, device=device) < self.p_transform
            indices_to_augment = torch.where(augment_mask)[0]
            num_to_augment = indices_to_augment.numel()

            if num_to_augment == 0:
                return time_series_batch

            # --- 🎲 Randomly sample Gaussian parameters for this call ---
            min_k, max_k = self.kernel_size_range
            kernel_size = torch.randint(min_k, max_k + 1, (1,)).item()
            kernel_size = kernel_size // 2 * 2 + 1  # Ensure kernel size is odd

            min_s, max_s = self.sigma_range
            sigma = (min_s + (max_s - min_s) * torch.rand(1)).item()

            # --- Get/Create Convolution Layers ---
            gauss_conv = self._create_gaussian_layer(kernel_size, sigma, num_channels, device)

            cache_key = (num_channels, device)
            if cache_key not in self.conv_cache:
                self.conv_cache[cache_key] = self._create_fixed_kernel_layers(num_channels, device)
            fixed_layers = self.conv_cache[cache_key]

            # --- Apply Augmentations ---
            subset_to_augment = time_series_batch[indices_to_augment]
            subset_permuted = subset_to_augment.permute(0, 2, 1)

            op_choices = torch.randint(0, 6, (num_to_augment,), device=device)

            smoothed_subset = gauss_conv(subset_permuted)
            sobel_on_smoothed = fixed_layers["sobel"](smoothed_subset)
            laplace_on_smoothed = fixed_layers["laplace"](smoothed_subset)
            d3_on_smoothed = fixed_layers["d3"](smoothed_subset)
            d4_on_smoothed = fixed_layers["d4"](smoothed_subset)

            gauss_result = self._rescale_signal(smoothed_subset, subset_permuted)
            sobel_result = self._rescale_signal(sobel_on_smoothed, subset_permuted)
            laplace_result = self._rescale_signal(laplace_on_smoothed, subset_permuted)
            d3_result = self._rescale_signal(d3_on_smoothed, subset_permuted)
            d4_result = self._rescale_signal(d4_on_smoothed, subset_permuted)

            use_right_integral = torch.rand(num_to_augment, 1, 1, device=device) > 0.5
            flipped_subset = torch.flip(subset_permuted, dims=[2])
            right_integral = torch.flip(torch.cumsum(flipped_subset, dim=2), dims=[2])
            left_integral = torch.cumsum(subset_permuted, dim=2)
            integral_result = torch.where(use_right_integral, right_integral, left_integral)
            integral_result_normalized = self._rescale_signal(integral_result, subset_permuted)

            # --- Assemble the results based on op_choices ---
            op_choices_view = op_choices.view(-1, 1, 1)
            augmented_subset = torch.where(op_choices_view == 0, gauss_result, subset_permuted)
            augmented_subset = torch.where(op_choices_view == 1, sobel_result, augmented_subset)
            augmented_subset = torch.where(op_choices_view == 2, laplace_result, augmented_subset)
            augmented_subset = torch.where(op_choices_view == 3, integral_result_normalized, augmented_subset)
            augmented_subset = torch.where(op_choices_view == 4, d3_result, augmented_subset)
            augmented_subset = torch.where(op_choices_view == 5, d4_result, augmented_subset)

            augmented_subset_final = augmented_subset.permute(0, 2, 1)
            augmented_batch = time_series_batch.clone()
            augmented_batch[indices_to_augment] = augmented_subset_final

            return augmented_batch


class RandomConvAugmenter:
    """
    Applies a stack of 1-to-N random 1D convolutions to a time series batch.

    This augmenter is inspired by the principles of ROCKET and RandConv,
    randomizing nearly every aspect of the convolution process to create a
    highly diverse set of transformations. This version includes multiple
    kernel generation strategies, random padding modes, and optional non-linearities.
    """

    def __init__(
        self,
        p_transform: float = 0.5,
        kernel_size_range: tuple[int, int] = (3, 31),
        dilation_range: tuple[int, int] = (1, 8),
        layer_range: tuple[int, int] = (1, 3),
        sigma_range: tuple[float, float] = (0.5, 5.0),
        bias_range: tuple[float, float] = (-0.5, 0.5),
    ):
        """
        Initializes the augmenter.

        Args:
            p_transform (float): Probability of applying the augmentation to a series.
            kernel_size_range (Tuple[int, int]): [min, max] range for kernel sizes.
                                                 Must be odd numbers.
            dilation_range (Tuple[int, int]): [min, max] range for dilation factors.
            layer_range (Tuple[int, int]): [min, max] range for the number of
                                           stacked convolution layers.
            sigma_range (Tuple[float, float]): [min, max] range for the sigma of
                                               Gaussian kernels.
            bias_range (Tuple[float, float]): [min, max] range for the bias term.
        """
        assert kernel_size_range[0] % 2 == 1 and kernel_size_range[1] % 2 == 1, "Kernel sizes must be odd."

        self.p_transform = p_transform
        self.kernel_size_range = kernel_size_range
        self.dilation_range = dilation_range
        self.layer_range = layer_range
        self.sigma_range = sigma_range
        self.bias_range = bias_range
        self.padding_modes = ["reflect", "replicate", "circular"]

    def _rescale_signal(self, processed_signal: torch.Tensor, original_signal: torch.Tensor) -> torch.Tensor:
        """Rescales the processed signal to match the min/max range of the original."""
        original_min = torch.amin(original_signal, dim=-1, keepdim=True)
        original_max = torch.amax(original_signal, dim=-1, keepdim=True)
        processed_min = torch.amin(processed_signal, dim=-1, keepdim=True)
        processed_max = torch.amax(processed_signal, dim=-1, keepdim=True)

        original_range = original_max - original_min
        processed_range = processed_max - processed_min
        epsilon = 1e-8

        is_flat = processed_range < epsilon

        rescaled_signal = (
            (processed_signal - processed_min) / (processed_range + epsilon)
        ) * original_range + original_min

        original_mean = torch.mean(original_signal, dim=-1, keepdim=True)
        flat_rescaled = original_mean.expand_as(original_signal)

        return torch.where(is_flat, flat_rescaled, rescaled_signal)

    def _apply_random_conv_stack(self, series: torch.Tensor) -> torch.Tensor:
        """
        Applies a randomly configured stack of convolutions to a single time series.

        Args:
            series (torch.Tensor): A single time series of shape (1, num_channels, seq_len).

        Returns:
            torch.Tensor: The augmented time series.
        """
        num_channels = series.shape[1]
        device = series.device

        num_layers = torch.randint(self.layer_range[0], self.layer_range[1] + 1, (1,)).item()

        processed_series = series
        for i in range(num_layers):
            # 1. Sample kernel size
            k_min, k_max = self.kernel_size_range
            kernel_size = torch.randint(k_min // 2, k_max // 2 + 1, (1,)).item() * 2 + 1

            # 2. Sample dilation
            d_min, d_max = self.dilation_range
            dilation = torch.randint(d_min, d_max + 1, (1,)).item()

            # 3. Sample bias
            b_min, b_max = self.bias_range
            bias_val = (b_min + (b_max - b_min) * torch.rand(1)).item()

            # 4. Sample padding mode
            padding_mode = np.random.choice(self.padding_modes)

            conv_layer = nn.Conv1d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding="same",  # Let PyTorch handle padding calculation
                padding_mode=padding_mode,
                groups=num_channels,
                bias=True,
                device=device,
            )

            # 5. Sample kernel weights from a wider variety of types
            weight_type = torch.randint(0, 4, (1,)).item()
            if weight_type == 0:  # Gaussian kernel
                s_min, s_max = self.sigma_range
                sigma = (s_min + (s_max - s_min) * torch.rand(1)).item()
                ax = torch.arange(
                    -(kernel_size // 2),
                    kernel_size // 2 + 1,
                    device=device,
                    dtype=torch.float32,
                )
                kernel = torch.exp(-0.5 * (ax / sigma) ** 2)
            elif weight_type == 1:  # Standard normal kernel
                kernel = torch.randn(kernel_size, device=device)
            elif weight_type == 2:  # Polynomial kernel
                coeffs = torch.randn(3, device=device)  # a, b, c for ax^2+bx+c
                x_vals = torch.linspace(-1, 1, kernel_size, device=device)
                kernel = coeffs[0] * x_vals**2 + coeffs[1] * x_vals + coeffs[2]
            else:  # Noisy Sobel kernel
                # Ensure kernel is large enough for a Sobel filter
                actual_kernel_size = 3 if kernel_size < 3 else kernel_size
                sobel_base = torch.tensor([-1, 0, 1], dtype=torch.float32, device=device)
                noise = torch.randn(3, device=device) * 0.1
                noisy_sobel = sobel_base + noise
                # Pad if the random kernel size is larger than 3
                pad_total = actual_kernel_size - 3
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left
                kernel = F.pad(noisy_sobel, (pad_left, pad_right), "constant", 0)

            # 6. Probabilistic normalization
            if torch.rand(1).item() < 0.8:  # 80% chance to normalize
                kernel /= torch.sum(torch.abs(kernel)) + 1e-8

            kernel = kernel.view(1, 1, -1).repeat(num_channels, 1, 1)

            conv_layer.weight.data = kernel
            conv_layer.bias.data.fill_(bias_val)
            conv_layer.weight.requires_grad = False
            conv_layer.bias.requires_grad = False

            # Apply convolution
            processed_series = conv_layer(processed_series)

            # 7. Optional non-linearity (not on the last layer)
            if i < num_layers - 1:
                activation_type = torch.randint(0, 3, (1,)).item()
                if activation_type == 1:
                    processed_series = F.relu(processed_series)
                elif activation_type == 2:
                    processed_series = torch.tanh(processed_series)
                # if 0, do nothing (linear)

        return processed_series

    def transform(self, time_series_batch: torch.Tensor) -> torch.Tensor:
        """Applies a random augmentation to a subset of the batch."""
        with torch.no_grad():
            if self.p_transform == 0:
                return time_series_batch

            batch_size, seq_len, num_channels = time_series_batch.shape
            device = time_series_batch.device

            augment_mask = torch.rand(batch_size, device=device) < self.p_transform
            indices_to_augment = torch.where(augment_mask)[0]
            num_to_augment = indices_to_augment.numel()

            if num_to_augment == 0:
                return time_series_batch

            subset_to_augment = time_series_batch[indices_to_augment]

            subset_permuted = subset_to_augment.permute(0, 2, 1)

            augmented_subset_list = []
            for i in range(num_to_augment):
                original_series = subset_permuted[i : i + 1]
                augmented_series = self._apply_random_conv_stack(original_series)

                rescaled_series = self._rescale_signal(augmented_series.squeeze(0), original_series.squeeze(0))
                augmented_subset_list.append(rescaled_series.unsqueeze(0))

            if augmented_subset_list:
                augmented_subset = torch.cat(augmented_subset_list, dim=0)
                augmented_subset_final = augmented_subset.permute(0, 2, 1)

                augmented_batch = time_series_batch.clone()
                augmented_batch[indices_to_augment] = augmented_subset_final
                return augmented_batch
            else:
                return time_series_batch
