import logging
import os
from typing import Optional

import torch

from src.data.containers import BatchTimeSeriesContainer
from src.data.utils import sample_future_length
from src.plotting.plot_multivariate_timeseries import plot_from_container
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
from src.synthetic_generation.gp_prior.gp_generator_wrapper import (
    GPGeneratorWrapper,
)
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
from src.synthetic_generation.steps.step_generator_wrapper import (
    StepGeneratorWrapper,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def visualize_batch_sample(
    generator,
    batch_size: int = 8,
    output_dir: str = "outputs/plots",
    sample_idx: Optional[int] = None,
    prefix: str = "",
    seed: Optional[int] = None,
) -> None:
    """
    Visualize a sample from a batch of synthetic multivariate time series from any generator.
    Also plot artificial predictions for demonstration if requested.

    Args:
        generator: Any generator wrapper (LMC, Kernel, GP, etc.)
        batch_size: Number of samples to generate in the batch
        output_dir: Directory to save plots
        sample_idx: Index of the sample to visualize
        seed: Seed for the generator
    """
    os.makedirs(output_dir, exist_ok=True)

    generator_name = generator.__class__.__name__
    logger.info(f"[{generator_name}] Generating batch of size {batch_size}")

    batch = generator.generate_batch(batch_size=batch_size, seed=seed)
    values = torch.from_numpy(batch.values)
    if values.ndim == 2:
        values = values.unsqueeze(-1)  # Add channel dimension: [batch_size, seq_len, 1]

    future_length = sample_future_length(range="gift_eval")
    # Slice along the time dimension (dimension 1)
    history_values = values[:, :-future_length, :]
    future_values = values[:, -future_length:, :]

    batch = BatchTimeSeriesContainer(
        history_values=history_values,
        future_values=future_values,
        start=batch.start,
        frequency=batch.frequency,
    )

    logger.info(
        f"[{generator_name}] Batch history values shape: {batch.history_values.shape}"
    )
    logger.info(
        f"[{generator_name}] Batch future values shape: {batch.future_values.shape}"
    )
    logger.info(f"[{generator_name}] Batch start: {batch.start}")
    logger.info(f"[{generator_name}] Batch frequency: {batch.frequency}")

    if sample_idx is None:
        for sample_idx in range(batch_size):
            filename = f"{prefix}_{generator_name.lower().replace('generatorwrapper', '')}_sample_{sample_idx}.png"
            output_file = os.path.join(output_dir, filename)
            title = f"{prefix.capitalize()} {generator_name.replace('GeneratorWrapper', '')} Synthetic Time Series (Sample {sample_idx})"

            plot_from_container(
                batch=batch,
                sample_idx=sample_idx,
                output_file=output_file,
                show=False,
                title=title,
            )
            logger.info(
                f"[{generator_name}] Saved plot for sample {sample_idx} to {output_file}"
            )
            logger.info("--------------------------------")


if __name__ == "__main__":
    # Configuration
    batch_size = 2
    total_length = 2048
    output_dir = "outputs/plots"
    global_seed = 2025

    logger.info(f"Saving plots to {output_dir}")

    kernel_params_univariate = KernelGeneratorParams(
        global_seed=global_seed,
        length=total_length,
    )
    kernel_gen_univariate = KernelGeneratorWrapper(kernel_params_univariate)

    gp_params_univariate = GPGeneratorParams(
        global_seed=global_seed,
        length=total_length,
    )
    gp_gen_univariate = GPGeneratorWrapper(gp_params_univariate)

    forecast_pfn_univariate_params = ForecastPFNGeneratorParams(
        global_seed=global_seed,
        length=total_length,
    )
    forecast_pfn_univariate_gen = ForecastPFNGeneratorWrapper(
        forecast_pfn_univariate_params
    )

    sine_wave_params = SineWaveGeneratorParams(
        global_seed=global_seed,
        length=total_length,
    )
    sine_wave_univariate_gen = SineWaveGeneratorWrapper(sine_wave_params)

    sawtooth_params = SawToothGeneratorParams(
        global_seed=global_seed,
        length=total_length,
    )
    sawtooth_univariate_gen = SawToothGeneratorWrapper(sawtooth_params)

    step_params = params = StepGeneratorParams(
        length=2048,
        global_seed=42,
    )
    step_gen_univariate = StepGeneratorWrapper(step_params)

    anomaly_params = AnomalyGeneratorParams(
        global_seed=global_seed,
        length=total_length,
    )
    anomaly_gen_univariate = AnomalyGeneratorWrapper(anomaly_params)

    spikes_params = SpikesGeneratorParams(
        global_seed=global_seed,
        length=total_length,
    )
    spikes_gen_univariate = SpikesGeneratorWrapper(spikes_params)

    cauker_params_multivariate = CauKerGeneratorParams(
        global_seed=global_seed,
        length=total_length,
        num_channels=5,
    )
    cauker_gen_multivariate = CauKerGeneratorWrapper(cauker_params_multivariate)

    ou_params = OrnsteinUhlenbeckProcessGeneratorParams(
        global_seed=global_seed,
        length=total_length,
    )
    ou_gen_univariate = OrnsteinUhlenbeckProcessGeneratorWrapper(ou_params)

    stochastic_rhythm_params = StochasticRhythmAudioParams(
        global_seed=global_seed,
        length=total_length,
    )
    stochastic_rhythm_gen_univariate = StochasticRhythmAudioWrapper(
        stochastic_rhythm_params
    )

    financial_volatility_params = FinancialVolatilityAudioParams(
        global_seed=global_seed,
        length=total_length,
    )
    financial_volatility_gen_univariate = FinancialVolatilityAudioWrapper(
        financial_volatility_params
    )

    multi_scale_fractal_params = MultiScaleFractalAudioParams(
        global_seed=global_seed,
        length=total_length,
    )
    multi_scale_fractal_gen_univariate = MultiScaleFractalAudioWrapper(
        multi_scale_fractal_params
    )

    network_topology_params = NetworkTopologyAudioParams(
        global_seed=global_seed,
        length=total_length,
    )
    network_topology_gen_univariate = NetworkTopologyAudioWrapper(
        network_topology_params
    )

    # Visualize samples from all generators
    visualize_batch_sample(
        kernel_gen_univariate, batch_size=batch_size, output_dir=output_dir
    )

    visualize_batch_sample(
        gp_gen_univariate, batch_size=batch_size, output_dir=output_dir
    )

    visualize_batch_sample(
        forecast_pfn_univariate_gen, batch_size=batch_size, output_dir=output_dir
    )

    visualize_batch_sample(
        sine_wave_univariate_gen,
        batch_size=batch_size,
        output_dir=output_dir,
    )

    visualize_batch_sample(
        sawtooth_univariate_gen,
        batch_size=batch_size,
        output_dir=output_dir,
    )

    visualize_batch_sample(
        step_gen_univariate,
        batch_size=batch_size,
        output_dir=output_dir,
    )

    visualize_batch_sample(
        anomaly_gen_univariate,
        batch_size=batch_size,
        output_dir=output_dir,
    )

    visualize_batch_sample(
        spikes_gen_univariate,
        batch_size=batch_size,
        output_dir=output_dir,
    )

    visualize_batch_sample(
        cauker_gen_multivariate,
        batch_size=batch_size,
        output_dir=output_dir,
        prefix="multivariate",
    )

    visualize_batch_sample(
        ou_gen_univariate,
        batch_size=batch_size,
        output_dir=output_dir,
        seed=global_seed,
    )

    visualize_batch_sample(
        stochastic_rhythm_gen_univariate, batch_size=batch_size, output_dir=output_dir
    )

    visualize_batch_sample(
        financial_volatility_gen_univariate,
        batch_size=batch_size,
        output_dir=output_dir,
    )

    visualize_batch_sample(
        multi_scale_fractal_gen_univariate,
        batch_size=batch_size,
        output_dir=output_dir,
    )

    visualize_batch_sample(
        network_topology_gen_univariate, batch_size=batch_size, output_dir=output_dir
    )
