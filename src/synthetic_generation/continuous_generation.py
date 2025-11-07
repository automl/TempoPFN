import argparse
import logging
import os
import random
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather

from src.synthetic_generation.anomalies.anomaly_generator_wrapper import (
    AnomalyGeneratorWrapper,
)
from src.synthetic_generation.audio_generators.financial_volatility_wrapper import (
    FinancialVolatilityAudioWrapper,
)
from src.synthetic_generation.audio_generators.multi_scale_fractal_wrapper import (
    MultiScaleFractalAudioWrapper,
)
from src.synthetic_generation.audio_generators.network_topology_wrapper import (
    NetworkTopologyAudioWrapper,
)
from src.synthetic_generation.audio_generators.stochastic_rhythm_wrapper import (
    StochasticRhythmAudioWrapper,
)
from src.synthetic_generation.cauker.cauker_generator_wrapper import (
    CauKerGeneratorWrapper,
)
from src.synthetic_generation.forecast_pfn_prior.forecast_pfn_generator_wrapper import (
    ForecastPFNGeneratorWrapper,
)
from src.synthetic_generation.generator_params import (
    AnomalyGeneratorParams,
    CauKerGeneratorParams,
    FinancialVolatilityAudioParams,
    ForecastPFNGeneratorParams,
    GPGeneratorParams,
    KernelGeneratorParams,
    MultiScaleFractalAudioParams,
    NetworkTopologyAudioParams,
    OrnsteinUhlenbeckProcessGeneratorParams,
    SawToothGeneratorParams,
    SineWaveGeneratorParams,
    SpikesGeneratorParams,
    StepGeneratorParams,
    StochasticRhythmAudioParams,
)
from src.synthetic_generation.gp_prior.gp_generator_wrapper import GPGeneratorWrapper
from src.synthetic_generation.kernel_synth.kernel_generator_wrapper import (
    KernelGeneratorWrapper,
)
from src.synthetic_generation.ornstein_uhlenbeck_process.ou_generator_wrapper import (
    OrnsteinUhlenbeckProcessGeneratorWrapper,
)
from src.synthetic_generation.sawtooth.sawtooth_generator_wrapper import (
    SawToothGeneratorWrapper,
)
from src.synthetic_generation.sine_waves.sine_wave_generator_wrapper import (
    SineWaveGeneratorWrapper,
)
from src.synthetic_generation.spikes.spikes_generator_wrapper import (
    SpikesGeneratorWrapper,
)
from src.synthetic_generation.steps.step_generator_wrapper import StepGeneratorWrapper


class TimeSeriesDatasetManager:
    """Manages writing time series data to disk in batches, safe for parallel runs."""

    def __init__(self, output_path: str, batch_size: int = 2**16):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.batches_dir = self.output_path
        self.batch_size = batch_size
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
        """Initializes state by scanning existing files to count total series."""
        existing_batches = sorted(self.batches_dir.glob("batch_*.arrow"))
        total_series = 0
        if not existing_batches:
            logging.info("No existing batches found. Starting from scratch.")
        else:
            for batch_file in existing_batches:
                try:
                    batch_table = feather.read_table(batch_file)
                    total_series += len(batch_table)
                except Exception as e:
                    logging.warning(f"Error reading {batch_file}: {e}, skipping.")
        self.series_counter = total_series
        logging.info(f"Found {self.series_counter} existing series in dataset.")

    def get_current_series_count(self) -> int:
        """Returns the total number of series found on disk at initialization."""
        return self.series_counter

    def append_batch(self, batch_data: list[dict[str, Any]]) -> None:
        """Appends a batch to a new file using an atomic rename for parallel safety."""
        if not batch_data:
            return

        try:
            arrays = []
            for field in self.schema:
                field_name = field.name
                if field_name in ["start", "generation_timestamp"]:
                    timestamps = [d[field_name] for d in batch_data]
                    arrays.append(pa.array([t.value for t in timestamps], type=pa.timestamp("ns")))
                else:
                    arrays.append(pa.array([d[field_name] for d in batch_data]))
            new_table = pa.Table.from_arrays(arrays, schema=self.schema)
        except Exception as e:
            logging.error(f"Error creating Arrow table: {e}")
            raise

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, dir=self.batches_dir, suffix=".arrow.tmp") as tmp:
                tmp_path = tmp.name
                feather.write_feather(new_table, tmp_path)

            max_retries = 20
            for _ in range(max_retries):
                existing = self.batches_dir.glob("batch_*.arrow")
                batch_nums = [int(p.stem.split("_")[1]) for p in existing if p.stem.split("_")[1].isdigit()]
                next_num = max(batch_nums) + 1 if batch_nums else 0
                target_path = self.batches_dir / f"batch_{next_num:08d}.arrow"
                try:
                    os.rename(tmp_path, target_path)
                    self.series_counter += len(batch_data)
                    logging.info(f"Saved {target_path.name} with {len(batch_data)} series.")
                    return
                except FileExistsError:
                    logging.warning(f"Race condition on {target_path.name}. Retrying...")
                    time.sleep(random.uniform(0.1, 1.0))

            raise OSError("Failed to write batch due to file conflicts.")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)


class GeneratorWrapper:
    def __init__(
        self,
        generator_type: str,
        length: int = 2048,
        global_seed: int = 42,
        num_channels: int | None = None,
    ):
        self.generator_type = generator_type
        self.length = length
        self.is_multivariate = generator_type.lower() in [
            "cauker_multivariate",
        ]
        self.explode_multivariate_to_univariate = generator_type.lower() == "cauker_univariate"
        self._explode_channels = 0

        # Create appropriate parameter object and wrapper
        if generator_type.lower() == "gp":
            params = GPGeneratorParams(
                global_seed=global_seed,
                length=length,
            )
            self.wrapper = GPGeneratorWrapper(params)
        elif generator_type.lower() == "kernel":
            params = KernelGeneratorParams(
                global_seed=global_seed,
                length=length,
            )
            self.wrapper = KernelGeneratorWrapper(params)
        elif generator_type.lower() == "forecast_pfn":
            params = ForecastPFNGeneratorParams(
                global_seed=global_seed,
                length=length,
                max_absolute_spread=500.0,
                max_absolute_value=500.0,
            )
            self.wrapper = ForecastPFNGeneratorWrapper(params)
        elif generator_type.lower() == "sinewave":
            params = SineWaveGeneratorParams(
                global_seed=global_seed,
                length=length,
            )
            self.wrapper = SineWaveGeneratorWrapper(params)
        elif generator_type.lower() == "sawtooth":
            params = SawToothGeneratorParams(
                global_seed=global_seed,
                length=length,
            )
            self.wrapper = SawToothGeneratorWrapper(params)
        elif generator_type.lower() == "cauker_univariate":
            params = CauKerGeneratorParams(
                global_seed=global_seed,
                length=length,
                num_channels=6,
            )
            self.wrapper = CauKerGeneratorWrapper(params)
            self._explode_channels = 6
        elif generator_type.lower() == "cauker_multivariate":
            effective_channels = (
                int(num_channels) if num_channels is not None else CauKerGeneratorParams().num_channels  # type: ignore[arg-type]
            )
            params = CauKerGeneratorParams(
                global_seed=global_seed,
                length=length,
                num_channels=effective_channels,
                num_nodes=effective_channels,
            )
            self.wrapper = CauKerGeneratorWrapper(params)
        elif generator_type.lower() == "step":
            params = StepGeneratorParams(
                global_seed=global_seed,
                length=length,
            )
            self.wrapper = StepGeneratorWrapper(params)
        elif generator_type.lower() == "spike":
            params = SpikesGeneratorParams(
                global_seed=global_seed,
                length=length,
            )
            self.wrapper = SpikesGeneratorWrapper(params)
        elif generator_type.lower() == "anomaly":
            params = AnomalyGeneratorParams(
                global_seed=global_seed,
                length=length,
            )
            self.wrapper = AnomalyGeneratorWrapper(params)
        elif generator_type.lower() == "ou_process":
            params = OrnsteinUhlenbeckProcessGeneratorParams(
                global_seed=global_seed,
                length=length,
            )
            self.wrapper = OrnsteinUhlenbeckProcessGeneratorWrapper(params)
        elif generator_type.lower() == "audio_financial_volatility":
            params = FinancialVolatilityAudioParams(
                global_seed=global_seed,
                length=length,
            )
            self.wrapper = FinancialVolatilityAudioWrapper(params)
        elif generator_type.lower() == "audio_multi_scale_fractal":
            params = MultiScaleFractalAudioParams(
                global_seed=global_seed,
                length=length,
            )
            self.wrapper = MultiScaleFractalAudioWrapper(params)
        elif generator_type.lower() == "audio_stochastic_rhythm":
            params = StochasticRhythmAudioParams(
                global_seed=global_seed,
                length=length,
            )
            self.wrapper = StochasticRhythmAudioWrapper(params)
        elif generator_type.lower() == "audio_network_topology":
            params = NetworkTopologyAudioParams(
                global_seed=global_seed,
                length=length,
            )
            self.wrapper = NetworkTopologyAudioWrapper(params)
        else:
            raise ValueError(f"Unsupported generator type: {generator_type}")

    def generate_batch(self, batch_size: int, start_seed: int) -> list[dict[str, Any]]:
        """Generate a batch of time series using the wrapper's batch generation."""
        try:
            if self.explode_multivariate_to_univariate and self._explode_channels > 0:
                base_batch_size = int(np.ceil(batch_size / self._explode_channels))
                container = self.wrapper.generate_batch(batch_size=base_batch_size, seed=start_seed)
            else:
                container = self.wrapper.generate_batch(batch_size=batch_size, seed=start_seed)

            batch_data = []
            container_batch_size = container.values.shape[0]
            for i in range(container_batch_size):
                series_id_base = start_seed + i

                if self.explode_multivariate_to_univariate:
                    series_data = container.values[i]
                    if series_data.ndim != 2:
                        raise ValueError("Expected multivariate data for CauKer univariate mode")
                    num_channels = series_data.shape[1]
                    for channel in range(num_channels):
                        channel_values = self._ensure_proper_format(series_data[:, channel])
                        values_list = [channel_values.tolist()]
                        batch_data.append(
                            {
                                "series_id": series_id_base * 1_000 + channel,
                                "values": values_list,
                                "length": len(channel_values),
                                "num_channels": 1,
                                "generator_type": self.generator_type,
                                "start": pd.Timestamp(container.start[i]),
                                "frequency": container.frequency[i].value,
                                "generation_timestamp": pd.Timestamp.now(),
                            }
                        )
                    continue
                elif self.is_multivariate:
                    series_data = container.values[i]
                    num_channels = series_data.shape[1]
                    values_list = [self._ensure_proper_format(series_data[:, c]).tolist() for c in range(num_channels)]
                    seq_length = len(values_list[0])
                else:
                    values = self._ensure_proper_format(container.values[i, :])
                    values_list = [values.tolist()]
                    num_channels = 1
                    seq_length = len(values)

                batch_data.append(
                    {
                        "series_id": series_id_base,
                        "values": values_list,
                        "length": seq_length,
                        "num_channels": num_channels,
                        "generator_type": self.generator_type,
                        "start": pd.Timestamp(container.start[i]),
                        "frequency": container.frequency[i].value,
                        "generation_timestamp": pd.Timestamp.now(),
                    }
                )

            if self.explode_multivariate_to_univariate:
                batch_data = batch_data[:batch_size]

            return batch_data

        except Exception as e:
            logging.error(f"Error generating batch: {e}")
            return []

    def _ensure_proper_format(self, values: Any) -> np.ndarray:
        values = np.asarray(values).flatten()
        if len(values) != self.length:
            logging.warning(f"Generated series length {len(values)} != expected {self.length}. Padding/truncating.")
            if len(values) > self.length:
                values = values[: self.length]
            else:
                values = np.pad(values, (0, self.length - len(values)), mode="constant")
        return values.astype(np.float64)


class ContinuousGenerator:
    def __init__(
        self,
        generator_wrapper: GeneratorWrapper,
        dataset_manager: TimeSeriesDatasetManager,
        batch_size: int = 2**16,
        run_id: int = 0,
    ):
        self.generator_wrapper = generator_wrapper
        self.dataset_manager = dataset_manager
        self.batch_size = batch_size
        self.run_id = run_id
        self.series_in_run = 0
        self.partial_batch_data: list[dict[str, Any]] = []
        self.shutting_down = False
        logging.info(f"Generator initialized for run_id: {self.run_id}")

    def _setup_signal_handlers(self) -> None:
        """Sets up signal handlers for graceful shutdown."""
        self.original_sigint = signal.getsignal(signal.SIGINT)
        self.original_sigterm = signal.getsignal(signal.SIGTERM)

        def graceful_shutdown(signum, frame):
            if self.shutting_down:
                return
            self.shutting_down = True
            logging.warning(f"\nSignal {signal.Signals(signum).name} received. Shutting down.")
            if self.partial_batch_data:
                logging.info(f"Saving incomplete batch of {len(self.partial_batch_data)} series...")
                try:
                    self.dataset_manager.append_batch(self.partial_batch_data)
                except Exception as e:
                    logging.error(f"Could not save partial batch on exit: {e}")
            sys.exit(0)

        signal.signal(signal.SIGINT, graceful_shutdown)
        signal.signal(signal.SIGTERM, graceful_shutdown)

    def run_continuous(self, num_batches_to_generate: int) -> None:
        """Runs the generation loop, creating chunks and saving batches."""
        self._setup_signal_handlers()
        logging.info(f"Job starting. Goal: {num_batches_to_generate} new batches.")
        start_time = time.time()
        batches_completed = 0

        while batches_completed < num_batches_to_generate:
            if self.shutting_down:
                logging.info("Shutdown signal caught, stopping generation.")
                break

            chunk_size = min(64, self.batch_size - len(self.partial_batch_data))

            # Create a seed that fits in uint32 range by combining run_id and series count
            # Use modulo to ensure it stays within valid range
            series_id_start = (self.run_id + self.series_in_run) % (2**32)

            new_chunk = self.generator_wrapper.generate_batch(batch_size=chunk_size, start_seed=series_id_start)

            if not new_chunk:
                logging.error("Generator failed to produce data. Stopping job.")
                break

            self.partial_batch_data.extend(new_chunk)
            self.series_in_run += len(new_chunk)

            if len(self.partial_batch_data) >= self.batch_size:
                batch_to_write = self.partial_batch_data[: self.batch_size]
                self.partial_batch_data = self.partial_batch_data[self.batch_size :]
                self.dataset_manager.append_batch(batch_to_write)
                batches_completed += 1

                elapsed = time.time() - start_time
                series_per_sec = (batches_completed * self.batch_size) / elapsed if elapsed > 0 else 0
                print(
                    f"✓ Completed batch {batches_completed}/{num_batches_to_generate} in job | "
                    f"Total Series in DS: {self.dataset_manager.series_counter:,} | "
                    f"Rate: {series_per_sec:.1f}/s"
                )

        if not self.shutting_down and self.partial_batch_data:
            logging.info(f"Job finished. Saving final partial batch of {len(self.partial_batch_data)}.")
            self.dataset_manager.append_batch(self.partial_batch_data)


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Continuous time series generation script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--generator",
        type=str,
        required=True,
        choices=[
            "forecast_pfn",
            "gp",
            "kernel",
            "cauker_univariate",
            "cauker_multivariate",
            "sinewave",
            "sawtooth",
            "step",
            "spike",
            "anomaly",
            "ou_process",
            "audio_financial_volatility",
            "audio_multi_scale_fractal",
            "audio_stochastic_rhythm",
            "audio_network_topology",
        ],
        help="Type of generator to use",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for datasets",
    )
    parser.add_argument("--length", type=int, default=2048, help="Length of each time series")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16384,
        help="Number of series per batch file",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=100,
        help="Number of batches to generate in this job run",
    )
    parser.add_argument(
        "--num-channels",
        type=int,
        help="Number of channels for multivariate generators (cauker_multivariate)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    setup_logging(args.verbose)

    # Use a high-precision timestamp for a unique run ID and a compatible seed
    run_id = time.time_ns()
    global_seed = run_id % (2**32)
    logging.info(f"Using unique run ID: {run_id} (Seed: {global_seed})")

    gen_name = args.generator.lower()
    if gen_name in ["cauker_multivariate"]:
        if args.num_channels is None or args.num_channels < 2:
            logging.error("--num-channels (>=2) is required for multivariate generators")
            sys.exit(2)
        dataset_dir_name = f"cauker_{args.num_channels}_variates"
    else:
        dataset_dir_name = args.generator

    output_path = Path(args.output_dir) / dataset_dir_name

    try:
        generator_wrapper = GeneratorWrapper(
            generator_type=args.generator,
            length=args.length,
            global_seed=global_seed,
            num_channels=args.num_channels,
        )
        dataset_manager = TimeSeriesDatasetManager(str(output_path), batch_size=args.batch_size)
        continuous_gen = ContinuousGenerator(
            generator_wrapper=generator_wrapper,
            dataset_manager=dataset_manager,
            batch_size=args.batch_size,
            run_id=run_id,
        )
        continuous_gen.run_continuous(num_batches_to_generate=args.num_batches)
        logging.info("Generation job completed successfully!")

    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
