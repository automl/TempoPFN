import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from src.data.augmentations import (
    CensorAugmenter,
    DifferentialAugmenter,
    MixUpAugmenter,
    QuantizationAugmenter,
    RandomConvAugmenter,
    TimeFlipAugmenter,
    YFlipAugmenter,
)
from src.data.constants import LENGTH_CHOICES
from src.data.datasets import CyclicalBatchDataset
from src.data.filter import is_low_quality
from src.data.scalers import MeanScaler, MedianScaler, MinMaxScaler, RobustScaler
from src.synthetic_generation.augmentations.offline_per_sample_iid_augmentations import (
    TimeSeriesDatasetManager,
    UnivariateOfflineAugmentor,
)


class OfflineTempBatchAugmentedGenerator:
    def __init__(
        self,
        base_data_dir: str,
        output_dir: str,
        length: int | None,
        mixed_batch_size: int = 10,
        chunk_size: int = 2**13,
        generator_proportions: dict[str, float] | None = None,
        augmentations: dict[str, bool] | None = None,
        augmentation_probabilities: dict[str, float] | None = None,
        global_seed: int = 42,
        mixup_position: str = "both",
        selection_strategy: str = "random",
        change_threshold: float = 0.05,
        enable_quality_filter: bool = False,
        temp_batch_retries: int = 3,
    ):
        self.base_data_dir = base_data_dir
        self.length = length
        self.mixed_batch_size = mixed_batch_size
        self.chunk_size = chunk_size
        self.global_seed = global_seed
        np.random.seed(global_seed)
        torch.manual_seed(global_seed)

        out_dir_name = f"augmented_temp_batch_{length}" if length is not None else "augmented_temp_batch"
        self.dataset_manager = TimeSeriesDatasetManager(str(Path(output_dir) / out_dir_name), batch_size=chunk_size)

        # Augmentation config
        self.augmentation_probabilities = augmentation_probabilities or {}
        self.augmentations = augmentations or {}
        self.apply_augmentations = any(self.augmentations.values())

        # RNG for category choices and sampling
        self.rng = np.random.default_rng(global_seed)

        # Mixup placement and selection strategy
        self.mixup_position = mixup_position
        self.selection_strategy = selection_strategy
        self.change_threshold = float(change_threshold)
        self.enable_quality_filter = bool(enable_quality_filter)
        self.temp_batch_retries = int(temp_batch_retries)

        # Initialize augmenters as in old composer ordering
        self.flip_augmenter = None
        if self.augmentations.get("time_flip_augmentation", False):
            self.flip_augmenter = TimeFlipAugmenter(
                p_flip=self.augmentation_probabilities.get("time_flip_augmentation", 0.5)
            )

        self.yflip_augmenter = None
        if self.augmentations.get("yflip_augmentation", False):
            self.yflip_augmenter = YFlipAugmenter(p_flip=self.augmentation_probabilities.get("yflip_augmentation", 0.5))

        self.censor_augmenter = None
        if self.augmentations.get("censor_augmentation", False):
            self.censor_augmenter = CensorAugmenter()

        self.quantization_augmenter = None
        if self.augmentations.get("quantization_augmentation", False):
            self.quantization_augmenter = QuantizationAugmenter(
                p_quantize=self.augmentation_probabilities.get("censor_or_quantization_augmentation", 0.5),
                level_range=(5, 15),
            )

        self.mixup_augmenter = None
        if self.augmentations.get("mixup_augmentation", False):
            self.mixup_augmenter = MixUpAugmenter(
                p_combine=self.augmentation_probabilities.get("mixup_augmentation", 0.5)
            )

        self.differential_augmentor = None
        if self.augmentations.get("differential_augmentation", False):
            self.differential_augmentor = DifferentialAugmenter(
                p_transform=self.augmentation_probabilities.get("differential_augmentation", 0.5)
            )

        self.random_conv_augmenter = None
        if self.augmentations.get("random_conv_augmentation", False):
            self.random_conv_augmenter = RandomConvAugmenter(
                p_transform=self.augmentation_probabilities.get("random_conv_augmentation", 0.3)
            )

        self.generator_proportions = self._setup_proportions(generator_proportions)
        self.datasets = self._initialize_datasets()

        # Per-series augmentor from offline_augmentations.py (categories only)
        self.per_series_augmentor = UnivariateOfflineAugmentor(
            augmentations=self.augmentations,
            augmentation_probabilities=self.augmentation_probabilities,
            global_seed=global_seed,
        )

    def _compute_change_scores(self, original_batch: torch.Tensor, augmented_batch: torch.Tensor) -> np.ndarray:
        # Normalized MAE vs IQR (q25-q75) per element
        bsz = augmented_batch.shape[0]
        scores: list[float] = []
        for i in range(bsz):
            base_flat = original_batch[i].reshape(-1)
            q25 = torch.quantile(base_flat, 0.25)
            q75 = torch.quantile(base_flat, 0.75)
            iqr = (q75 - q25).item()
            iqr = iqr if iqr > 0 else 1e-6
            mae = torch.mean(torch.abs(augmented_batch[i] - original_batch[i])).item()
            scores.append(mae / iqr)
        return np.asarray(scores, dtype=float)

    def _setup_proportions(self, generator_proportions: dict[str, float] | None) -> dict[str, float]:
        # Default uniform across discovered generators
        if generator_proportions is None:
            base = Path(self.base_data_dir)
            discovered = [p.name for p in base.iterdir() if p.is_dir()]
            proportions = dict.fromkeys(discovered, 1.0)
        else:
            proportions = dict(generator_proportions)

        total = sum(proportions.values())
        if total <= 0:
            raise ValueError("Total generator proportions must be positive")
        return {k: v / total for k, v in proportions.items()}

    def _initialize_datasets(self) -> dict[str, CyclicalBatchDataset]:
        datasets: dict[str, CyclicalBatchDataset] = {}
        for generator_name, proportion in self.generator_proportions.items():
            if proportion <= 0:
                continue
            batches_dir = Path(self.base_data_dir) / generator_name
            if not batches_dir.is_dir():
                logging.warning(f"Skipping '{generator_name}' because directory does not exist: {batches_dir}")
                continue
            try:
                dataset = CyclicalBatchDataset(
                    batches_dir=str(batches_dir),
                    generator_type=generator_name,
                    device=None,
                    prefetch_next=True,
                    prefetch_threshold=32,
                )
                datasets[generator_name] = dataset
                logging.info(f"Loaded dataset for {generator_name}")
            except Exception as e:
                logging.warning(f"Failed to load dataset for {generator_name}: {e}")
        if not datasets:
            raise ValueError("No valid datasets loaded from base_data_dir")
        return datasets

    def _sample_generator_name(self) -> str:
        available = [g for g in self.generator_proportions.keys() if g in self.datasets]
        probs = np.array([self.generator_proportions[g] for g in available], dtype=float)
        probs = probs / probs.sum()
        return str(self.rng.choice(available, p=probs))

    def _series_key(self, gen_name: str, sample: dict, values: torch.Tensor) -> str:
        series_id = sample.get("series_id", None)
        if series_id is not None:
            return f"{gen_name}:{series_id}"
        # Fallback: hash by values and metadata
        try:
            arr = values.detach().cpu().numpy()
            h = hash(
                (
                    gen_name,
                    sample.get("start", None),
                    sample.get("frequency", None),
                    arr.shape,
                    float(arr.mean()),
                    float(arr.std()),
                )
            )
            return f"{gen_name}:hash:{h}"
        except Exception:
            return f"{gen_name}:rand:{self.rng.integers(0, 1 << 31)}"

    def _convert_sample_to_tensor(self, sample: dict) -> tuple[torch.Tensor, pd.Timestamp, str, int]:
        num_channels = sample.get("num_channels", 1)
        values_data = sample["values"]

        if num_channels == 1:
            if isinstance(values_data[0], list):
                values = torch.tensor(values_data[0], dtype=torch.float32)
            else:
                values = torch.tensor(values_data, dtype=torch.float32)
            values = values.unsqueeze(0).unsqueeze(-1)
        else:
            channel_tensors = []
            for channel_values in values_data:
                channel_tensor = torch.tensor(channel_values, dtype=torch.float32)
                channel_tensors.append(channel_tensor)
            values = torch.stack(channel_tensors, dim=-1).unsqueeze(0)

        freq_str = sample["frequency"]
        start_val = sample["start"]
        start = start_val if isinstance(start_val, pd.Timestamp) else pd.Timestamp(start_val)
        return values, start, freq_str, num_channels

    def _shorten_like_batch_composer(self, values: torch.Tensor, target_len: int) -> torch.Tensor | None:
        # Only shorten if longer; if shorter than target_len, reject (to keep batch aligned)
        seq_len = int(values.shape[1])
        if seq_len == target_len:
            return values
        if seq_len < target_len:
            return None
        # Randomly choose cut or subsample with equal probability
        strategy = str(self.rng.choice(["cut", "subsample"]))
        if strategy == "cut":
            max_start_idx = seq_len - target_len
            start_idx = int(self.rng.integers(0, max_start_idx + 1))
            return values[:, start_idx : start_idx + target_len, :]
        # Subsample evenly spaced indices
        indices = np.linspace(0, seq_len - 1, target_len, dtype=int)
        return values[:, indices, :]

    def _maybe_apply_scaler(self, values: torch.Tensor) -> torch.Tensor:
        scaler_choice = str(self.rng.choice(["robust", "minmax", "median", "mean", "none"]))
        scaler = None
        if scaler_choice == "robust":
            scaler = RobustScaler()
        elif scaler_choice == "minmax":
            scaler = MinMaxScaler()
        elif scaler_choice == "median":
            scaler = MedianScaler()
        elif scaler_choice == "mean":
            scaler = MeanScaler()
        if scaler is not None:
            values = scaler.scale(values, scaler.compute_statistics(values))
        return values

    def _apply_augmentations(
        self,
        batch_values: torch.Tensor,
        starts: list[pd.Timestamp],
        freqs: list[str],
    ) -> torch.Tensor:
        if not self.apply_augmentations:
            return batch_values

        # 1) Early mixup (batch-level)
        if (
            self.mixup_position in ["first", "both"]
            and self.augmentations.get("mixup_augmentation", False)
            and self.mixup_augmenter is not None
        ):
            batch_values = self.mixup_augmenter.transform(batch_values)

        # 2) Per-series categories (apply to ALL series with starts/freqs)
        batch_size = int(batch_values.shape[0])
        augmented_list = []
        for i in range(batch_size):
            s = batch_values[i : i + 1]
            start_i = starts[i] if i < len(starts) else None
            freq_i = freqs[i] if i < len(freqs) else None
            s_aug = self.per_series_augmentor.apply_per_series_only(s, start=start_i, frequency=freq_i)
            augmented_list.append(s_aug)
        batch_values = torch.cat(augmented_list, dim=0)

        # 3) Noise augmentation (batch-level)
        if self.augmentations.get("noise_augmentation", False):
            if self.rng.random() < self.augmentation_probabilities.get("noise_augmentation", 0.5):
                noise_std = 0.01 * torch.std(batch_values)
                if torch.isfinite(noise_std) and (noise_std > 0):
                    noise = torch.normal(0, noise_std, size=batch_values.shape)
                    batch_values = batch_values + noise

        # 4) Scaling augmentation (batch-level)
        if self.augmentations.get("scaling_augmentation", False):
            if self.rng.random() < self.augmentation_probabilities.get("scaling_augmentation", 0.5):
                scale_factor = float(self.rng.uniform(0.95, 1.05))
                batch_values = batch_values * scale_factor

        # 5) RandomConvAugmenter (batch-level)
        if self.augmentations.get("random_conv_augmentation", False) and self.random_conv_augmenter is not None:
            if self.rng.random() < self.augmentation_probabilities.get("random_conv_augmentation", 0.3):
                batch_values = self.random_conv_augmenter.transform(batch_values)

        # 6) Late mixup (batch-level)
        if (
            self.mixup_position in ["last", "both"]
            and self.augmentations.get("mixup_augmentation", False)
            and self.mixup_augmenter is not None
        ):
            batch_values = self.mixup_augmenter.transform(batch_values)

        return batch_values

    def _get_one_source_sample(
        self, total_length_for_batch: int, used_source_keys: set
    ) -> tuple[torch.Tensor, pd.Timestamp, str, str] | None:
        # Returns (values, start, freq, source_key) or None if cannot fetch
        attempts = 0
        while attempts < 50:
            attempts += 1
            gen_name = self._sample_generator_name()
            dataset = self.datasets[gen_name]
            sample = dataset.get_samples(1)[0]
            values, start, freq_str, num_channels = self._convert_sample_to_tensor(sample)
            if num_channels != 1:
                continue
            # Reject NaNs
            if torch.isnan(values).any():
                continue
            # Shorten to target_len; reject if too short
            shortened = self._shorten_like_batch_composer(values, total_length_for_batch)
            if shortened is None:
                continue
            values = shortened
            # Random scaler
            values = self._maybe_apply_scaler(values)
            # Uniqueness check
            key = self._series_key(gen_name, sample, values)
            if key in used_source_keys:
                continue
            # Reserve key immediately to avoid re-use in same temp batch
            used_source_keys.add(key)
            return values, start, freq_str, key
        return None

    def _tensor_to_values_list(self, series_tensor: torch.Tensor) -> tuple[list[list[float]], int, int]:
        seq_len = int(series_tensor.shape[1])
        num_channels = int(series_tensor.shape[2])
        if num_channels == 1:
            return [series_tensor.squeeze(0).squeeze(-1).tolist()], seq_len, 1
        channels: list[list[float]] = []
        for ch in range(num_channels):
            channels.append(series_tensor[0, :, ch].tolist())
        return channels, seq_len, num_channels

    def run(self, num_batches: int) -> None:
        logging.info(
            f"Starting offline IID augmentation into {self.dataset_manager.batches_dir} | "
            f"chunk_size={self.chunk_size} | "
            f"mixed_batch_size={self.mixed_batch_size}"
        )

        augmented_buffer: list[dict[str, Any]] = []
        target_batches = num_batches
        start_time = time.time()

        try:
            while self.dataset_manager.batch_counter < target_batches:
                # Decide target length for this temp batch
                total_length_for_batch = (
                    self.length if self.length is not None else int(self.rng.choice(LENGTH_CHOICES))
                )

                selected_record: dict[str, Any] | None = None
                for _retry in range(max(1, self.temp_batch_retries + 1)):
                    # Collect a temporary mixed batch without reusing sources
                    temp_values_list: list[torch.Tensor] = []
                    temp_starts: list[pd.Timestamp] = []
                    temp_freqs: list[str] = []
                    temp_used_keys: set = set()

                    attempts = 0
                    while len(temp_values_list) < self.mixed_batch_size and attempts < self.mixed_batch_size * 200:
                        attempts += 1
                        fetched = self._get_one_source_sample(total_length_for_batch, temp_used_keys)
                        if fetched is None:
                            continue
                        values, start, freq, _ = fetched
                        temp_values_list.append(values)
                        temp_starts.append(start)
                        temp_freqs.append(freq)

                    if len(temp_values_list) == 0:
                        # If we could not fetch anything, rebuild next retry
                        continue

                    temp_batch = torch.cat(temp_values_list, dim=0)
                    original_temp_batch = temp_batch.clone()

                    # Apply augmentations sequentially
                    augmented_temp_batch = self._apply_augmentations(temp_batch, temp_starts, temp_freqs)

                    # Compute change scores
                    scores = self._compute_change_scores(original_temp_batch, augmented_temp_batch)

                    # Build eligible indices by threshold
                    eligible = np.where(scores >= self.change_threshold)[0].tolist()

                    # Apply quality filter if enabled
                    if self.enable_quality_filter:
                        eligible_q: list[int] = []
                        for idx in eligible:
                            cand = augmented_temp_batch[idx : idx + 1]
                            if not is_low_quality(cand):
                                eligible_q.append(idx)
                        eligible = eligible_q

                    sel_idx: int | None = None
                    if self.selection_strategy == "max_change":
                        if eligible:
                            sel_idx = int(max(eligible, key=lambda i: scores[i]))
                        else:
                            # Fallback to best by score (respect quality if possible)
                            if self.enable_quality_filter:
                                qual_idxs = [
                                    i
                                    for i in range(augmented_temp_batch.shape[0])
                                    if not is_low_quality(augmented_temp_batch[i : i + 1])
                                ]
                                if qual_idxs:
                                    sel_idx = int(max(qual_idxs, key=lambda i: scores[i]))
                            if sel_idx is None:
                                sel_idx = int(np.argmax(scores))
                    else:
                        # random selection among eligible, else fallback to best
                        if eligible:
                            sel_idx = int(self.rng.choice(np.asarray(eligible, dtype=int)))
                        else:
                            if self.enable_quality_filter:
                                qual_idxs = [
                                    i
                                    for i in range(augmented_temp_batch.shape[0])
                                    if not is_low_quality(augmented_temp_batch[i : i + 1])
                                ]
                                if qual_idxs:
                                    sel_idx = int(max(qual_idxs, key=lambda i: scores[i]))
                            if sel_idx is None:
                                sel_idx = int(np.argmax(scores))

                    # If still none (shouldn't happen), rebuild
                    if sel_idx is None:
                        continue

                    selected_series = augmented_temp_batch[sel_idx : sel_idx + 1]
                    values_list, seq_len, num_channels = self._tensor_to_values_list(selected_series)
                    selected_record = {
                        "series_id": self.dataset_manager.series_counter,
                        "values": values_list,
                        "length": int(seq_len),
                        "num_channels": int(num_channels),
                        "generator_type": "augmented",
                        "start": pd.Timestamp(temp_starts[sel_idx]),
                        "frequency": temp_freqs[sel_idx],
                        "generation_timestamp": pd.Timestamp.now(),
                    }
                    break

                if selected_record is None:
                    # Could not assemble a valid candidate after retries; skip iteration
                    continue

                augmented_buffer.append(selected_record)

                if len(augmented_buffer) >= self.chunk_size:
                    write_start = time.time()
                    self.dataset_manager.append_batch(augmented_buffer)
                    write_time = time.time() - write_start
                    elapsed = time.time() - start_time
                    series_per_sec = self.dataset_manager.series_counter / elapsed if elapsed > 0 else 0
                    print(
                        f"✓ Wrote batch {self.dataset_manager.batch_counter - 1}/{target_batches} | "
                        f"Series: {self.dataset_manager.series_counter:,} | "
                        f"Rate: {series_per_sec:.1f}/s | "
                        f"Write: {write_time:.2f}s"
                    )
                    augmented_buffer = []

        except KeyboardInterrupt:
            logging.info(
                f"Interrupted. Generated {self.dataset_manager.series_counter} series, "
                f"{self.dataset_manager.batch_counter} batches."
            )
        finally:
            if augmented_buffer:
                self.dataset_manager.append_batch(augmented_buffer)
            logging.info("Offline IID augmentation completed.")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Offline IID augmentation script using temp mixed batches",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--base-data-dir",
        type=str,
        required=True,
        help="Base directory with generator subdirectories (inputs)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Base output directory for augmented datasets",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=None,
        help="Fixed length of augmented series. If set, saves under augmented{length}",
    )
    parser.add_argument(
        "--mixed-batch-size",
        type=int,
        default=64,
        help="Temporary mixed batch size before selecting a single element",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2**13,
        help="Number of series per written Arrow batch",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=1000,
        help="Number of Arrow batches to write",
    )
    parser.add_argument(
        "--mixup-position",
        type=str,
        default="both",
        choices=["first", "last", "both"],
        help="Where to apply mixup in the pipeline (first, last, or both)",
    )
    parser.add_argument(
        "--selection-strategy",
        type=str,
        default="random",
        choices=["random", "max_change"],
        help="How to select the final series from the temp batch",
    )
    parser.add_argument(
        "--change-threshold",
        type=float,
        default=0.05,
        help="Minimum normalized change score (vs IQR) required for selection",
    )
    parser.add_argument(
        "--enable-quality-filter",
        action="store_true",
        help="Enable low-quality filter using autocorr/SNR/complexity",
    )
    parser.add_argument(
        "--temp-batch-retries",
        type=int,
        default=3,
        help="Number of times to rebuild temp batch if selection fails thresholds",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--global-seed", type=int, default=42, help="Global random seed")

    args = parser.parse_args()
    setup_logging(args.verbose)

    generator_proportions = {
        "forecast_pfn": 1.0,
        "gp": 1.0,
        "kernel": 1.0,
        "sinewave": 1.0,
        "sawtooth": 1.0,
        "step": 0.1,
        "anomaly": 1.0,
        "spike": 1.0,
        "cauker_univariate": 2.0,
        "ou_process": 1.0,
        "audio_financial_volatility": 0.1,
        "audio_multi_scale_fractal": 0.1,
        "audio_network_topology": 0.5,
        "audio_stochastic_rhythm": 1.0,
    }

    # Defaults reflecting configs/train.yaml from the prompt
    augmentations = {
        "censor_augmentation": True,
        "quantization_augmentation": False,
        "mixup_augmentation": True,
        "time_flip_augmentation": True,
        "yflip_augmentation": True,
        "differential_augmentation": True,
        "regime_change_augmentation": True,
        "shock_recovery_augmentation": True,
        "calendar_augmentation": False,
        "amplitude_modulation_augmentation": True,
        "resample_artifacts_augmentation": True,
        "scaling_augmentation": True,
        "noise_augmentation": True,
        "random_conv_augmentation": True,
    }

    augmentation_probabilities = {
        "censor_or_quantization_augmentation": 0.40,
        "mixup_augmentation": 0.50,
        "time_flip_augmentation": 0.30,
        "yflip_augmentation": 0.30,
        "differential_augmentation": 0.40,
        "regime_change_augmentation": 0.40,
        "shock_recovery_augmentation": 0.40,
        "calendar_augmentation": 0.40,
        "amplitude_modulation_augmentation": 0.35,
        "resample_artifacts_augmentation": 0.40,
        "scaling_augmentation": 0.50,
        "noise_augmentation": 0.10,
        "random_conv_augmentation": 0.30,
    }

    try:
        generator = OfflineTempBatchAugmentedGenerator(
            base_data_dir=args.base_data_dir,
            output_dir=args.output_dir,
            length=args.length,
            mixed_batch_size=args.mixed_batch_size,
            chunk_size=args.chunk_size,
            generator_proportions=generator_proportions,
            augmentations=augmentations,
            augmentation_probabilities=augmentation_probabilities,
            global_seed=args.global_seed,
            mixup_position=args.mixup_position,
            selection_strategy=args.selection_strategy,
            change_threshold=args.change_threshold,
            enable_quality_filter=args.enable_quality_filter,
            temp_batch_retries=args.temp_batch_retries,
        )

        generator.run(num_batches=args.num_batches)
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
