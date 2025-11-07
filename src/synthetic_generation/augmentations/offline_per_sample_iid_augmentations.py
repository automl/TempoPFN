import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
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
from src.data.frequency import Frequency, parse_frequency
from src.data.scalers import MeanScaler, MedianScaler, MinMaxScaler, RobustScaler


class TimeSeriesDatasetManager:
    def __init__(self, output_path: str, batch_size: int = 2**13):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.batches_dir = self.output_path
        self.batches_dir.mkdir(exist_ok=True)
        self.batch_size = batch_size
        self.batch_counter = 0
        self.series_counter = 0

        self.schema = pa.schema(
            [
                ("series_id", pa.int64()),
                ("values", pa.list_(pa.list_(pa.float64()))),
                ("length", pa.int32()),
                ("num_channels", pa.int32()),
                ("generator_type", pa.string()),
                ("start", pa.timestamp("ns")),
                ("frequency", pa.string()),
                ("generation_timestamp", pa.timestamp("ns")),
            ]
        )

        self._initialize_state()

    def _initialize_state(self) -> None:
        existing_batches = sorted(self.batches_dir.glob("batch_*.arrow"))
        if not existing_batches:
            logging.info("No existing batches found. Starting from scratch.")
            return

        batch_numbers = []
        total_series = 0
        for batch_file in existing_batches:
            try:
                batch_num = int(batch_file.stem.split("_")[1])
                batch_numbers.append(batch_num)
                batch_table = feather.read_table(batch_file)
                total_series += len(batch_table)
            except Exception as e:
                logging.warning(f"Error reading batch {batch_file}: {e}")
                continue

        if batch_numbers:
            max_batch_num = max(batch_numbers)
            self.batch_counter = max_batch_num + 1
            self.series_counter = total_series

            last_batch_file = self.batches_dir / f"batch_{max_batch_num:08d}.arrow"
            if last_batch_file.exists():
                try:
                    last_batch_table = feather.read_table(last_batch_file)
                    if len(last_batch_table) < self.batch_size:
                        self.batch_counter = max_batch_num
                        logging.info(f"Found incomplete last batch {max_batch_num} with {len(last_batch_table)} series")
                except Exception as e:
                    logging.warning(f"Error checking last batch: {e}")

        logging.info(f"Resuming from: batch_counter={self.batch_counter}, series_counter={self.series_counter}")

    def append_batch(self, batch_data: list[dict[str, Any]]) -> None:
        if not batch_data:
            return

        try:
            arrays = []
            for field in self.schema:
                field_name = field.name
                if field_name in ["start", "generation_timestamp"]:
                    timestamps = [row[field_name] for row in batch_data]
                    arrays.append(pa.array([ts.value for ts in timestamps], type=pa.timestamp("ns")))
                else:
                    arrays.append(pa.array([row[field_name] for row in batch_data]))

            new_table = pa.Table.from_arrays(arrays, schema=self.schema)
            batch_filename = f"batch_{self.batch_counter:08d}.arrow"
            batch_filepath = self.batches_dir / batch_filename
            feather.write_feather(new_table, batch_filepath)

            self.series_counter += len(batch_data)
            self.batch_counter += 1

        except Exception as e:
            logging.error(f"Error writing batch: {e}")
            raise


class UnivariateOfflineAugmentor:
    def __init__(
        self,
        augmentations: dict[str, bool] | None = None,
        augmentation_probabilities: dict[str, float] | None = None,
        global_seed: int = 42,
    ):
        self.global_seed = global_seed
        np.random.seed(global_seed)
        torch.manual_seed(global_seed)
        self.rng = np.random.default_rng(global_seed)
        self.augmentation_probabilities = augmentation_probabilities
        self.augmentations = augmentations
        self.apply_augmentations = any(self.augmentations.values())

        self.time_flip_augmenter = None
        if self.augmentations["time_flip_augmentation"]:
            self.time_flip_augmenter = TimeFlipAugmenter(
                p_flip=self.augmentation_probabilities["time_flip_augmentation"]
            )

        self.yflip_augmenter = None
        if self.augmentations["yflip_augmentation"]:
            self.yflip_augmenter = YFlipAugmenter(p_flip=self.augmentation_probabilities["yflip_augmentation"])

        self.censor_augmenter = None
        if self.augmentations["censor_augmentation"]:
            self.censor_augmenter = CensorAugmenter()

        self.quantization_augmenter = None
        if self.augmentations["quantization_augmentation"]:
            self.quantization_augmenter = QuantizationAugmenter(
                p_quantize=self.augmentation_probabilities["censor_or_quantization_augmentation"],
                level_range=(5, 15),
            )

        if self.augmentations["differential_augmentation"]:
            self.differential_augmentor = DifferentialAugmenter(
                p_transform=self.augmentation_probabilities["differential_augmentation"]
            )

    def apply(
        self,
        history_values: torch.Tensor,
        starts: list[pd.Timestamp] | None = None,
        frequencies: list[str] | None = None,
    ) -> torch.Tensor:
        if not self.apply_augmentations:
            return history_values

        batch_size = int(history_values.shape[0])

        # 0) Combination (MixUp) – handled early at batch level due to dependency on other series
        if self.augmentations.get("mixup_augmentation", False) and self.mixup_augmenter is not None:
            history_values = self.mixup_augmenter.transform(history_values)

        # Per-series plan: sample categories and apply in fixed order per series
        # Categories (max one op per category):
        # invariances, structure, seasonality, artifacts, analytic, discrete
        for b in range(batch_size):
            series = history_values[b : b + 1].clone()

            # Determine eligible categories and weights for this series
            categories = [
                "invariances",
                "structure",
                "seasonality",
                "artifacts",
                "analytic",
                "discrete",
            ]
            weights = {
                "invariances": 0.6,
                "structure": 0.6,
                "seasonality": 0.5,
                "artifacts": 0.3,
                "analytic": 0.4,
                "discrete": 0.6,
            }

            # Remove disabled categories
            if not (
                self.augmentations.get("time_flip_augmentation", False)
                or self.augmentations.get("yflip_augmentation", False)
            ):
                weights["invariances"] = 0.0
            if not (
                self.augmentations.get("regime_change_augmentation", False)
                or self.augmentations.get("shock_recovery_augmentation", False)
            ):
                weights["structure"] = 0.0
            if not (
                self.augmentations.get("calendar_augmentation", False)
                or self.augmentations.get("amplitude_modulation_augmentation", False)
            ):
                weights["seasonality"] = 0.0
            if not self.augmentations.get("differential_augmentation", False):
                weights["analytic"] = 0.0
            if not (
                self.augmentations.get("quantization_augmentation", False)
                or self.augmentations.get("censor_augmentation", False)
            ):
                weights["discrete"] = 0.0

            # Sample number of operations in [2, 5]
            num_ops = int(self.rng.integers(2, 6))

            # Build candidate list and normalized probabilities
            candidates = [c for c in categories if weights[c] > 0.0]
            if not candidates:
                # Nothing to do for this series
                history_values[b : b + 1] = series
                continue
            num_ops = min(num_ops, len(candidates))
            probs = np.array([weights[c] for c in candidates], dtype=float)
            probs = probs / probs.sum()
            chosen_categories = list(self.rng.choice(candidates, size=num_ops, replace=False, p=probs))

            # Apply in the fixed global order, only if selected
            # 1) Invariances
            if "invariances" in chosen_categories:
                # Choose one: time_flip or yflip
                choices = []
                if self.augmentations.get("time_flip_augmentation", False):
                    choices.append("time_flip")
                if self.augmentations.get("yflip_augmentation", False):
                    choices.append("yflip")
                if choices:
                    pick = str(self.rng.choice(choices))
                    if pick == "time_flip":
                        series = torch.flip(series, dims=[1])
                    elif pick == "yflip":
                        series = -series

            # 2) Structural edits
            if "structure" in chosen_categories:
                choices = []
                if self.augmentations.get("regime_change_augmentation", False):
                    choices.append("regime")
                if self.augmentations.get("shock_recovery_augmentation", False):
                    choices.append("shock")
                if choices:
                    pick = str(self.rng.choice(choices))
                    if pick == "regime":
                        series = self._apply_regime_change(series, p_apply=1.0)
                    else:
                        series = self._apply_shock_recovery(series, p_apply=1.0)

            # 3) Seasonality/context
            if "seasonality" in chosen_categories:
                choices = []
                if self.augmentations.get("calendar_augmentation", False):
                    choices.append("calendar")
                if self.augmentations.get("amplitude_modulation_augmentation", False):
                    choices.append("amplitude")
                if choices:
                    pick = str(self.rng.choice(choices))
                    if pick == "calendar":
                        series = self._apply_calendar_injections(
                            series,
                            [starts[b]] if (starts is not None and b < len(starts)) else None,
                            [frequencies[b]] if (frequencies is not None and b < len(frequencies)) else None,
                            p_apply=1.0,
                        )
                    else:
                        series = self._apply_seasonality_amplitude_modulation(series, p_apply=1.0)

            # 4) Sampling artifacts
            if "artifacts" in chosen_categories and self.augmentations.get("resample_artifacts_augmentation", False):
                series = self._apply_resample_artifacts(series, p_apply=1.0)

            # 5) Analytic transforms
            if (
                "analytic" in chosen_categories
                and self.augmentations.get("differential_augmentation", False)
                and hasattr(self, "differential_augmentor")
            ):
                series = self.differential_augmentor.transform(series)

            # 6) Discretization/clipping (mutually exclusive)
            if "discrete" in chosen_categories:
                can_quant = (
                    self.augmentations.get("quantization_augmentation", False)
                    and self.quantization_augmenter is not None
                )
                can_cens = self.augmentations.get("censor_augmentation", False) and self.censor_augmenter is not None
                if can_quant and can_cens:
                    method = self.rng.choice(["quantize", "censor"], p=[0.6, 0.4])
                    if method == "quantize":
                        series = self.quantization_augmenter.transform(series)
                    else:
                        series = self.censor_augmenter.transform(series)
                elif can_quant:
                    series = self.quantization_augmenter.transform(series)
                elif can_cens:
                    series = self.censor_augmenter.transform(series)

            # Write back series
            history_values[b : b + 1] = series

        # 7) Scaling then Noise (last, optional, batch-level)
        if self.augmentations.get("scaling_augmentation", False):
            if self.rng.random() < self.augmentation_probabilities.get("scaling_augmentation", 0.0):
                scale_factor = float(self.rng.uniform(0.95, 1.05))
                history_values = history_values * scale_factor

        if self.augmentations.get("noise_augmentation", False):
            if self.rng.random() < self.augmentation_probabilities.get("noise_augmentation", 0.0):
                noise_std = 0.01 * torch.std(history_values)
                if torch.isfinite(noise_std) and (noise_std > 0):
                    noise = torch.normal(0, noise_std, size=history_values.shape)
                    history_values = history_values + noise

        return history_values

    def apply_per_series_only(
        self,
        series: torch.Tensor,
        start: pd.Timestamp | None = None,
        frequency: str | None = None,
    ) -> torch.Tensor:
        """
        Apply all per-series augmentations (excluding mixup) to a single series tensor,
        preserving ordering and probabilities used in apply().

        Args:
            series: Tensor of shape [1, length, 1]
            start: Optional pandas.Timestamp for calendar injections
            frequency: Optional frequency string for calendar injections
        """
        if not self.apply_augmentations:
            return series

        categories = [
            "invariances",
            "structure",
            "seasonality",
            "artifacts",
            "analytic",
            "discrete",
        ]
        weights = {
            "invariances": 0.6,
            "structure": 0.6,
            "seasonality": 0.5,
            "artifacts": 0.3,
            "analytic": 0.4,
            "discrete": 0.6,
        }

        # Disable categories not enabled
        if not (
            self.augmentations.get("time_flip_augmentation", False)
            or self.augmentations.get("yflip_augmentation", False)
        ):
            weights["invariances"] = 0.0
        if not (
            self.augmentations.get("regime_change_augmentation", False)
            or self.augmentations.get("shock_recovery_augmentation", False)
        ):
            weights["structure"] = 0.0
        if not (
            self.augmentations.get("calendar_augmentation", False)
            or self.augmentations.get("amplitude_modulation_augmentation", False)
        ):
            weights["seasonality"] = 0.0
        if not self.augmentations.get("differential_augmentation", False):
            weights["analytic"] = 0.0
        if not (
            self.augmentations.get("quantization_augmentation", False)
            or self.augmentations.get("censor_augmentation", False)
        ):
            weights["discrete"] = 0.0

        # Sample number of operations in [2, 5]
        num_ops = int(self.rng.integers(2, 6))
        candidates = [c for c in categories if weights[c] > 0.0]
        if not candidates:
            result = series
        else:
            num_ops = min(num_ops, len(candidates))
            probs = np.array([weights[c] for c in candidates], dtype=float)
            probs = probs / probs.sum()
            chosen_categories = list(self.rng.choice(candidates, size=num_ops, replace=False, p=probs))

            result = series.clone()

            # 1) Invariances
            if "invariances" in chosen_categories:
                choices = []
                if self.augmentations.get("time_flip_augmentation", False):
                    choices.append("time_flip")
                if self.augmentations.get("yflip_augmentation", False):
                    choices.append("yflip")
                if choices:
                    pick = str(self.rng.choice(choices))
                    if pick == "time_flip":
                        result = torch.flip(result, dims=[1])
                    elif pick == "yflip":
                        result = -result

            # 2) Structural edits
            if "structure" in chosen_categories:
                choices = []
                if self.augmentations.get("regime_change_augmentation", False):
                    choices.append("regime")
                if self.augmentations.get("shock_recovery_augmentation", False):
                    choices.append("shock")
                if choices:
                    pick = str(self.rng.choice(choices))
                    if pick == "regime":
                        result = self._apply_regime_change(result, p_apply=1.0)
                    else:
                        result = self._apply_shock_recovery(result, p_apply=1.0)

            # 3) Seasonality/context
            if "seasonality" in chosen_categories:
                choices = []
                if self.augmentations.get("calendar_augmentation", False):
                    choices.append("calendar")
                if self.augmentations.get("amplitude_modulation_augmentation", False):
                    choices.append("amplitude")
                if choices:
                    pick = str(self.rng.choice(choices))
                    if pick == "calendar":
                        result = self._apply_calendar_injections(
                            result,
                            [start] if start is not None else None,
                            [frequency] if frequency is not None else None,
                            p_apply=1.0,
                        )
                    else:
                        result = self._apply_seasonality_amplitude_modulation(result, p_apply=1.0)

            # 4) Sampling artifacts
            if "artifacts" in chosen_categories and self.augmentations.get("resample_artifacts_augmentation", False):
                result = self._apply_resample_artifacts(result, p_apply=1.0)

            # 5) Analytic transforms
            if (
                "analytic" in chosen_categories
                and self.augmentations.get("differential_augmentation", False)
                and hasattr(self, "differential_augmentor")
            ):
                result = self.differential_augmentor.transform(result)

            # 6) Discretization/clipping (mutually exclusive)
            if "discrete" in chosen_categories:
                can_quant = (
                    self.augmentations.get("quantization_augmentation", False)
                    and self.quantization_augmenter is not None
                )
                can_cens = self.augmentations.get("censor_augmentation", False) and self.censor_augmenter is not None
                if can_quant and can_cens:
                    method = self.rng.choice(["quantize", "censor"], p=[0.6, 0.4])
                    if method == "quantize":
                        result = self.quantization_augmenter.transform(result)
                    else:
                        result = self.censor_augmenter.transform(result)
                elif can_quant:
                    result = self.quantization_augmenter.transform(result)
                elif can_cens:
                    result = self.censor_augmenter.transform(result)

        # Optional scaling and noise (applied to this single series)
        if self.augmentations.get("scaling_augmentation", False):
            if self.rng.random() < self.augmentation_probabilities.get("scaling_augmentation", 0.0):
                scale_factor = float(self.rng.uniform(0.95, 1.05))
                result = result * scale_factor

        if self.augmentations.get("noise_augmentation", False):
            if self.rng.random() < self.augmentation_probabilities.get("noise_augmentation", 0.0):
                noise_std = 0.01 * torch.std(result)
                if torch.isfinite(noise_std) and (noise_std > 0):
                    noise = torch.normal(0, noise_std, size=result.shape)
                    result = result + noise

        return result

    @property
    def mixup_augmenter(self) -> MixUpAugmenter | None:
        if not hasattr(self, "_mixup_augmenter"):
            self._mixup_augmenter = (
                MixUpAugmenter(p_combine=self.augmentation_probabilities["mixup_augmentation"])
                if self.augmentations["mixup_augmentation"]
                else None
            )
        return self._mixup_augmenter

    def _apply_regime_change(self, series: torch.Tensor, p_apply: float) -> torch.Tensor:
        """
        Apply piecewise affine transforms with 1-3 change-points per series.
        series shape: [batch, length, 1]
        """
        if series.numel() == 0:
            return series
        batch_size, length, _ = series.shape
        result = series.clone()

        # Iterate per-series to allow different change-points
        for b in range(batch_size):
            if self.rng.random() >= p_apply:
                continue
            # sample number of change points and ensure minimum segment length
            num_cp = int(self.rng.integers(1, 4))
            min_seg = max(8, length // 32)
            if length <= (num_cp + 1) * min_seg:
                num_cp = max(1, length // (2 * min_seg) - 1)
            if num_cp <= 0:
                num_cp = 1
            # pick change-point indices
            valid_positions = np.arange(min_seg, length - min_seg)
            if valid_positions.size == 0:
                continue
            cp = np.sort(self.rng.choice(valid_positions, size=num_cp, replace=False))
            boundaries = np.concatenate([[0], cp, [length]])

            # compute per-segment scale/shift
            series_b = result[b, :, 0]
            seg_scales = []
            seg_shifts = []
            overall_std = torch.std(series_b).item()
            if not np.isfinite(overall_std) or overall_std == 0:
                overall_std = 1.0
            for _ in range(len(boundaries) - 1):
                scale = float(self.rng.uniform(0.8, 1.25))
                shift = float(self.rng.normal(0.0, 0.15 * overall_std))
                seg_scales.append(scale)
                seg_shifts.append(shift)

            # apply per segment
            for i in range(len(boundaries) - 1):
                s, e = int(boundaries[i]), int(boundaries[i + 1])
                if e <= s:
                    continue
                segment = series_b[s:e]
                # preserve segment mean roughly while scaling deviations
                seg_mean = torch.mean(segment)
                transformed = (segment - seg_mean) * seg_scales[i] + seg_mean + seg_shifts[i]
                result[b, s:e, 0] = transformed
        return result

    def _apply_shock_recovery(self, series: torch.Tensor, p_apply: float) -> torch.Tensor:
        """
        Add an impulse at a random time and exponentially decay to baseline.
        series shape: [batch, length, 1]
        """
        if series.numel() == 0:
            return series
        batch_size, length, _ = series.shape
        device = series.device
        result = series.clone()

        time_idx = torch.arange(length, device=device).float()

        for b in range(batch_size):
            if self.rng.random() >= p_apply:
                continue
            # choose shock time away from edges
            t0 = int(self.rng.integers(low=max(1, length // 16), high=max(2, length - length // 16)))
            # magnitude relative to series std
            s_b = result[b, :, 0]
            std_b = torch.std(s_b).item()
            if not np.isfinite(std_b) or std_b == 0:
                std_b = 1.0
            mag = float(self.rng.uniform(0.5, 2.0) * std_b)
            if self.rng.random() < 0.5:
                mag = -mag
            # decay constant
            half_life = float(self.rng.uniform(0.03, 0.25) * length)
            decay = torch.exp(-(time_idx - t0).clamp(min=0) / max(1.0, half_life))
            effect = mag * decay
            result[b, :, 0] = s_b + effect
        return result

    def _apply_calendar_injections(
        self,
        series: torch.Tensor,
        starts: list[pd.Timestamp] | None,
        frequencies: list[str] | None,
        p_apply: float,
    ) -> torch.Tensor:
        if series.numel() == 0:
            return series
        if starts is None or frequencies is None:
            return series
        batch_size, length, _ = series.shape
        result = series.clone()

        for b in range(batch_size):
            if b >= len(starts) or b >= len(frequencies):
                continue
            if self.rng.random() >= p_apply:
                continue
            start_ts = starts[b]
            try:
                freq_enum = parse_frequency(str(frequencies[b]))
                freq_alias = freq_enum.to_pandas_freq(for_date_range=True)
            except Exception:
                freq_alias = "D"
            try:
                index = pd.date_range(start=start_ts, periods=length, freq=freq_alias)
            except Exception:
                index = pd.date_range(start=start_ts, periods=length, freq="D")

            factors = np.ones(length, dtype=np.float32)
            # Weekend dips (for daily/hourly-like)
            try:
                freq_enum_check = parse_frequency(str(frequencies[b]))
            except Exception:
                freq_enum_check = Frequency.D
            if freq_enum_check in [
                Frequency.H,
                Frequency.D,
                Frequency.S,
                Frequency.T1,
                Frequency.T5,
                Frequency.T10,
                Frequency.T15,
                Frequency.T30,
            ]:
                dow = index.dayofweek
                if (dow >= 5).any():
                    dip = float(self.rng.uniform(0.7, 0.95))
                    factors[dow >= 5] *= dip

            # Month-end bumps
            if hasattr(index, "is_month_end"):
                me = np.asarray(index.is_month_end, dtype=bool)
                if me.any():
                    bump = float(self.rng.uniform(1.05, 1.3))
                    factors[me] *= bump

            # Holiday-like one-off effects (1-2 random impulses)
            n_imp = int(self.rng.integers(1, 3))
            imp_positions = self.rng.integers(0, length, size=n_imp)
            for pos in np.atleast_1d(imp_positions):
                if 0 <= pos < length:
                    impulse = float(self.rng.uniform(0.8, 1.4))
                    factors[pos] *= impulse

            # Apply multiplicatively around mean to avoid drift
            s = result[b, :, 0].cpu().numpy()
            mean_val = float(np.mean(s))
            s_new = (s - mean_val) * factors + mean_val
            result[b, :, 0] = torch.from_numpy(s_new).to(result.device)
        return result

    def _apply_seasonality_amplitude_modulation(self, series: torch.Tensor, p_apply: float) -> torch.Tensor:
        if series.numel() == 0:
            return series
        batch_size, length, _ = series.shape
        result = series.clone()

        for b in range(batch_size):
            if self.rng.random() >= p_apply:
                continue
            min_w = max(8, length // 16)
            max_w = max(min_w + 1, length // 2)
            win = int(self.rng.integers(min_w, max_w + 1))
            start = int(self.rng.integers(0, max(1, length - win)))
            end = start + win
            seg = result[b, start:end, 0]
            if seg.numel() == 0:
                continue
            seg_mean = torch.mean(seg)
            amp = float(self.rng.uniform(0.5, 1.8))
            result[b, start:end, 0] = (seg - seg_mean) * amp + seg_mean
        return result

    def _apply_resample_artifacts(
        self,
        series: torch.Tensor,
        p_apply: float,
    ) -> torch.Tensor:
        """
        Downsample then upsample with interpolation to introduce artifacts.
        """
        if series.numel() == 0:
            return series
        batch_size, length, _ = series.shape
        result = series.clone()

        for b in range(batch_size):
            if self.rng.random() >= p_apply:
                continue

            s_np = result[b, :, 0].cpu().numpy()
            max_factor = max(2, min(8, length // 32))
            if max_factor <= 1:
                continue
            factor = int(self.rng.integers(2, max_factor + 1))
            offset = int(self.rng.integers(0, factor))
            ds_idx = np.arange(offset, length, factor)
            if ds_idx.size < 3:
                continue
            ds_vals = s_np[ds_idx]
            base_idx = np.arange(length)
            mode = self.rng.choice(["linear", "hold", "linear_smooth"], p=[0.5, 0.2, 0.3])
            if mode == "linear":
                us = np.interp(base_idx, ds_idx, ds_vals)
            elif mode == "hold":
                us = np.empty(length, dtype=s_np.dtype)
                last = ds_vals[0]
                j = 0
                for i in range(length):
                    while j + 1 < ds_idx.size and i >= ds_idx[j + 1]:
                        j += 1
                        last = ds_vals[j]
                    us[i] = last
            else:
                us = np.interp(base_idx, ds_idx, ds_vals)
                k = max(3, length // 128)
                kernel = np.ones(k) / k
                us = np.convolve(us, kernel, mode="same")
            result[b, :, 0] = torch.from_numpy(us).to(result.device)
        return result


class OfflinePerSampleAugmentedGenerator:
    def __init__(
        self,
        base_data_dir: str,
        output_dir: str,
        length: int | None,
        chunk_size: int = 2**13,
        generator_proportions: dict[str, float] | None = None,
        augmentations: dict[str, bool] | None = None,
        augmentation_probabilities: dict[str, float] | None = None,
        global_seed: int = 42,
        mixup_position: str = "both",
        change_threshold: float = 0.05,
        max_tries: int = 3,
        enable_quality_filter: bool = False,
        rc_batch_size: int = 8,
    ):
        self.base_data_dir = base_data_dir
        self.length = length
        self.chunk_size = chunk_size
        self.global_seed = global_seed
        np.random.seed(global_seed)
        torch.manual_seed(global_seed)
        self.rng = np.random.default_rng(global_seed)
        self.mixup_position = mixup_position
        self.change_threshold = float(change_threshold)
        self.max_tries = int(max_tries)
        self.enable_quality_filter = bool(enable_quality_filter)
        self.rc_batch_size = int(rc_batch_size)

        out_dir_name = f"augmented_per_sample_{length}" if length is not None else "augmented_per_sample"
        self.dataset_manager = TimeSeriesDatasetManager(str(Path(output_dir) / out_dir_name), batch_size=chunk_size)

        self.augmentor = UnivariateOfflineAugmentor(
            augmentations=augmentations,
            augmentation_probabilities=augmentation_probabilities,
            global_seed=global_seed,
        )

        self.generator_proportions = self._setup_proportions(generator_proportions)
        self.datasets = self._initialize_datasets()

    # -------------------- Per-sample scaler utilities --------------------
    def _choose_scaler(self) -> object | None:
        """Choose a scaler with 50% probability of None; else one of four scalers uniformly."""
        if self.rng.random() < 0.5:
            return None
        pick = str(self.rng.choice(["robust", "minmax", "median", "mean"]))
        if pick == "robust":
            return RobustScaler()
        if pick == "minmax":
            return MinMaxScaler()
        if pick == "median":
            return MedianScaler()
        return MeanScaler()

    def _apply_scaler(self, values: torch.Tensor, scaler: object | None) -> torch.Tensor:
        """Apply the provided scaler to values of shape [1, length, channels]."""
        if scaler is None:
            return values
        stats = scaler.compute_statistics(values)
        return scaler.scale(values, stats)

    # -------------------- Mixup utilities (per-sample) --------------------
    def _mix_sources_static(self, source_tensor: torch.Tensor, alpha: float) -> torch.Tensor:
        """Static Dirichlet mix of k sources -> [1, L, C]."""
        k = int(source_tensor.shape[0])
        device = source_tensor.device
        concentration = torch.full((k,), float(alpha), device=device)
        weights = torch.distributions.Dirichlet(concentration).sample()
        mixed = (source_tensor * weights.view(k, 1, 1)).sum(dim=0, keepdim=True)
        return mixed

    def _apply_mixup_to_series(
        self,
        base_series: torch.Tensor,
        total_length_for_batch: int,
        scaler: object | None,
    ) -> torch.Tensor:
        """Mix base with k-1 additional sources; returns [1, L, 1]."""
        mixup = self.augmentor.mixup_augmenter
        if mixup is None:
            return base_series

        # Decide k
        current_k = mixup._sample_k() if not mixup.randomize_k else int(self.rng.integers(2, mixup.max_k + 1))
        # Ensure at least 2 and include base in the set
        current_k = max(2, int(current_k))
        num_sources_needed = current_k - 1

        chosen_gens = self._choose_generators_for_mixup(current_k)
        # If we sampled k gens but need only k-1 external sources, trim
        chosen_gens = chosen_gens[:num_sources_needed]

        sources: list[torch.Tensor] = []
        # Base (already possibly scaled) first
        sources.append(base_series)
        # Additional sources
        for gen in chosen_gens:
            src_values, _, _, _ = self._get_one_sample_from_generator(gen, total_length_for_batch)
            if scaler is not None:
                src_values = self._apply_scaler(src_values, scaler)
            sources.append(src_values)

        source_tensor = torch.cat(sources, dim=0)
        alpha = mixup._sample_alpha()
        mixed_series = self._mix_sources_static(source_tensor, alpha=alpha)
        return mixed_series

    # -------------------- RandomConv (temp batch) utilities --------------------
    def _apply_random_conv_with_temp_batch(
        self,
        base_series: torch.Tensor,
        total_length_for_batch: int,
        scaler: object | None,
    ) -> torch.Tensor:
        """Apply RandomConvAugmenter by creating a small temp batch and taking the transformed base element."""
        if not hasattr(self, "random_conv_augmenter"):
            # Lazy init if not present but enabled in config
            if self.augmentor.augmentations.get("random_conv_augmentation", False):
                p_val = self.augmentor.augmentation_probabilities.get("random_conv_augmentation", 0.3)
                self.random_conv_augmenter = RandomConvAugmenter(p_transform=p_val)
            else:
                return base_series

        # Assemble temp batch: base + (rc_batch_size-1) sources
        temp_series_list: list[torch.Tensor] = [base_series]
        for _ in range(max(0, self.rc_batch_size - 1)):
            try:
                gen = self._sample_generator_name()
                src_values, _, _, _ = self._get_one_sample_from_generator(gen, total_length_for_batch)
                if scaler is not None:
                    src_values = self._apply_scaler(src_values, scaler)
                temp_series_list.append(src_values)
            except Exception:
                break
        temp_batch = torch.cat(temp_series_list, dim=0)

        transformed = self.random_conv_augmenter.transform(temp_batch)
        return transformed[0:1]

    # -------------------- Selection and quality helpers --------------------
    def _compute_change_score(self, original: torch.Tensor, augmented: torch.Tensor) -> float:
        """
        Computes a normalized change score between original and augmented series.
        The score is the Mean Absolute Error (MAE) normalized by a robust
        measure of the original series' scale (its Interquartile Range).
        This makes the score less sensitive to outliers and absolute scale.
        """
        original_flat = original.flatten()

        # Use the standard Interquartile Range (IQR) for robust scaling.
        q25 = torch.quantile(original_flat, 0.25)
        q75 = torch.quantile(original_flat, 0.75)
        iqr = (q75 - q25).item()

        # Use a robust epsilon to prevent division by zero for flat series.
        series_range = (torch.max(original_flat) - torch.min(original_flat)).item()
        scale = max(iqr, 1e-6 * series_range, 1e-8)

        # Compute Mean Absolute Error
        mae = torch.mean(torch.abs(augmented - original)).item()

        return float(mae / scale)

    # moved to src/synthetic_generation/augmentations/filter.py

    def _setup_proportions(self, generator_proportions: dict[str, float] | None) -> dict[str, float]:
        # Default uniform proportions across discovered generators
        if generator_proportions is None:
            # Discover generator directories
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
            # Load batches only if the generator is explicitly listed and has positive proportion
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

    def _convert_sample_to_tensor(self, sample: dict) -> tuple[torch.Tensor, Any, str, int]:
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
        # Keep start as pandas.Timestamp for Arrow writing later
        if isinstance(start_val, pd.Timestamp):
            start = start_val
        else:
            start = pd.Timestamp(start_val)

        return values, start, freq_str, num_channels

    def _maybe_resize(self, values: torch.Tensor, target_len: int) -> torch.Tensor:
        if values.shape[1] == target_len:
            return values
        if values.shape[1] > target_len:
            max_start_idx = values.shape[1] - target_len
            start_idx = np.random.randint(0, max_start_idx + 1)
            return values[:, start_idx : start_idx + target_len, :]
        # Subsample evenly to reach target_len
        indices = np.linspace(0, values.shape[1] - 1, target_len, dtype=int)
        return values[:, indices, :]

    def _sample_generator_name(self) -> str:
        available = [g for g in self.generator_proportions.keys() if g in self.datasets]
        probs = np.array([self.generator_proportions[g] for g in available], dtype=float)
        probs = probs / probs.sum()
        return str(np.random.choice(available, p=probs))

    def _get_one_sample(self, total_length_for_batch: int) -> tuple[torch.Tensor, pd.Timestamp, str, int]:
        attempts = 0
        while attempts < 20:
            attempts += 1
            gen_name = self._sample_generator_name()
            dataset = self.datasets[gen_name]
            sample = dataset.get_samples(1)[0]
            values, start, freq_str, num_channels = self._convert_sample_to_tensor(sample)
            values = self._maybe_resize(values, total_length_for_batch)
            if values.shape[2] != 1:
                continue
            return values, start, freq_str, num_channels
        raise RuntimeError("Failed to sample a valid univariate series after multiple attempts")

    def _get_one_sample_from_generator(
        self, gen_name: str, total_length_for_batch: int
    ) -> tuple[torch.Tensor, pd.Timestamp, str, int]:
        attempts = 0
        dataset = self.datasets[gen_name]
        while attempts < 20:
            attempts += 1
            sample = dataset.get_samples(1)[0]
            values, start, freq_str, num_channels = self._convert_sample_to_tensor(sample)
            values = self._maybe_resize(values, total_length_for_batch)
            if values.shape[2] != 1:
                continue
            return values, start, freq_str, num_channels
        raise RuntimeError(
            f"Failed to sample a valid univariate series from generator '{gen_name}' after multiple attempts"
        )

    def _choose_generators_for_mixup(self, k: int) -> list[str]:
        available = [g for g in self.generator_proportions.keys() if g in self.datasets]
        if not available:
            raise RuntimeError("No available generators to sample from for mixup")
        k_eff = min(k, len(available))
        # Weighted sampling without replacement by sequential renormalization
        chosen: list[str] = []
        remaining = available.copy()
        while len(chosen) < k_eff:
            weights = np.array([self.generator_proportions[g] for g in remaining], dtype=float)
            if weights.sum() <= 0:
                # fallback to uniform
                probs = np.ones(len(remaining)) / len(remaining)
            else:
                probs = weights / weights.sum()
            pick = str(np.random.choice(remaining, p=probs))
            chosen.append(pick)
            remaining.remove(pick)
        return chosen

    def _maybe_apply_mixup_to_single(self, base_series: torch.Tensor, total_length_for_batch: int) -> torch.Tensor:
        do_mixup = self.augmentor.augmentations.get(
            "mixup_augmentation", False
        ) and self.augmentor.rng.random() < self.augmentor.augmentation_probabilities.get("mixup_augmentation", 0.0)
        if not do_mixup:
            return base_series

        # Use MixUpAugmenter to avoid duplication
        mixup = self.augmentor.mixup_augmenter
        if mixup is None:
            return base_series

        # Decide number of sources k consistent with MixUpAugmenter behavior
        current_k = mixup._sample_k() if not mixup.randomize_k else int(self.augmentor.rng.integers(2, mixup.max_k + 1))

        # Choose distinct generators for sources according to proportions
        chosen_gens = self._choose_generators_for_mixup(current_k)

        # Collect one source per chosen generator
        sources: list[torch.Tensor] = []
        for gen in chosen_gens:
            src_values, _, _, _ = self._get_one_sample_from_generator(gen, total_length_for_batch)
            sources.append(src_values)
        source_tensor = torch.cat(sources, dim=0)

        # Sample alpha via MixUpAugmenter, then mix
        alpha = mixup._sample_alpha()
        mixed_series = mixup.mix_sources(source_tensor, alpha=alpha)
        return mixed_series

    def _tensor_to_values_list(self, series_tensor: torch.Tensor) -> tuple[list[list[float]], int, int]:
        # series_tensor shape: [1, seq_len, num_channels]
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
            f"Starting offline augmentation into {self.dataset_manager.batches_dir} | chunk_size={self.chunk_size}"
        )

        augmented_buffer: list[dict[str, Any]] = []
        target_batches = num_batches
        start_time = time.time()

        try:
            while self.dataset_manager.batch_counter < target_batches:
                # Decide target length for this sample
                total_length_for_batch = (
                    self.length if self.length is not None else int(np.random.choice(LENGTH_CHOICES))
                )

                for _ in range(max(1, self.max_tries)):
                    # Sample one base series
                    base_values, base_start, base_freq, _ = self._get_one_sample(total_length_for_batch)
                    original_base = base_values.clone()

                    # Per-sample scaler choice (50% none; else robust/minmax/median/mean)
                    per_sample_scaler = self._choose_scaler()
                    base_values = self._apply_scaler(base_values, per_sample_scaler)

                    # Early mixup (if enabled and position includes first)
                    do_mixup_early = (
                        self.augmentor.augmentations.get("mixup_augmentation", False)
                        and self.mixup_position in ["first", "both"]
                        and self.augmentor.rng.random()
                        < self.augmentor.augmentation_probabilities.get("mixup_augmentation", 0.0)
                    )
                    if do_mixup_early:
                        base_values = self._apply_mixup_to_series(
                            base_values, total_length_for_batch, per_sample_scaler
                        )

                    # Apply per-series augmentations
                    augmented_single = self.augmentor.apply_per_series_only(
                        base_values, start=base_start, frequency=base_freq
                    )

                    # Optional analytic: RandomConvAugmenter via temp batch (before late mixup)
                    if self.augmentor.augmentations.get("random_conv_augmentation", False):
                        if self.rng.random() < self.augmentor.augmentation_probabilities.get(
                            "random_conv_augmentation", 0.3
                        ):
                            augmented_single = self._apply_random_conv_with_temp_batch(
                                augmented_single,
                                total_length_for_batch,
                                per_sample_scaler,
                            )

                    # Late mixup (if enabled and position includes last)
                    do_mixup_late = (
                        self.augmentor.augmentations.get("mixup_augmentation", False)
                        and self.mixup_position in ["last", "both"]
                        and self.augmentor.rng.random()
                        < self.augmentor.augmentation_probabilities.get("mixup_augmentation", 0.0)
                    )
                    if do_mixup_late:
                        augmented_single = self._apply_mixup_to_series(
                            augmented_single, total_length_for_batch, per_sample_scaler
                        )

                    # Compute change score and unchanged check
                    score = self._compute_change_score(original_base, augmented_single)
                    if score < self.change_threshold:
                        continue

                    # Optional quality filter
                    if self.enable_quality_filter and is_low_quality(augmented_single):
                        continue

                    # Accept first candidate that passes thresholds
                    values_list, seq_len, num_channels = self._tensor_to_values_list(augmented_single)
                    record = {
                        "series_id": self.dataset_manager.series_counter,
                        "values": values_list,
                        "length": int(seq_len),
                        "num_channels": int(num_channels),
                        "generator_type": "augmented",
                        "start": pd.Timestamp(base_start),
                        "frequency": base_freq,
                        "generation_timestamp": pd.Timestamp.now(),
                    }
                    augmented_buffer.append(record)
                    break

                # Discard combined_values_augmented and loop
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
            # Flush remaining buffer if any
            if augmented_buffer:
                self.dataset_manager.append_batch(augmented_buffer)
            logging.info("Offline augmentation completed.")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Offline augmentation script to precompute augmented series",
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
        "--chunk-size",
        type=int,
        default=2**13,  # 8192
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
        "--change-threshold",
        type=float,
        default=0.05,
        help="Minimum normalized change score (vs IQR) required to keep series",
    )
    parser.add_argument(
        "--max-tries",
        type=int,
        default=3,
        help="Max attempts to produce an acceptable augmented series per output",
    )
    parser.add_argument(
        "--enable-quality-filter",
        action="store_true",
        help="Enable low-quality series filter (noise-like removal)",
    )
    # Quality filter thresholds moved to filter module defaults
    parser.add_argument(
        "--rc-batch-size",
        type=int,
        default=8,
        help="Temporary batch size used for RandomConvAugmenter",
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

    augmentations = {
        "censor_augmentation": True,
        "quantization_augmentation": True,
        "mixup_augmentation": True,
        "time_flip_augmentation": True,
        "yflip_augmentation": True,
        "differential_augmentation": True,
        "regime_change_augmentation": True,
        "shock_recovery_augmentation": True,
        "calendar_augmentation": True,
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
        "noise_augmentation": 0.1,
        "random_conv_augmentation": 0.30,
    }

    try:
        generator = OfflinePerSampleAugmentedGenerator(
            base_data_dir=args.base_data_dir,
            output_dir=args.output_dir,
            length=args.length,
            chunk_size=args.chunk_size,
            generator_proportions=generator_proportions,
            augmentations=augmentations,
            augmentation_probabilities=augmentation_probabilities,
            global_seed=args.global_seed,
            mixup_position=args.mixup_position,
            change_threshold=args.change_threshold,
            max_tries=args.max_tries,
            enable_quality_filter=args.enable_quality_filter,
            rc_batch_size=args.rc_batch_size,
        )

        generator.run(num_batches=args.num_batches)
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
