import logging
import os
from typing import List, Optional

import torch

from src.data.containers import BatchTimeSeriesContainer
from src.data.utils import sample_future_length
from src.plotting.plot_timeseries import plot_from_container
from src.synthetic_generation.anomalies.anomaly_generator_wrapper import (
    AnomalyGeneratorWrapper,
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

PYO_AVAILABLE = True
try:
    import pyo  # requires portaudio to be installed
except (ImportError, OSError):
    PYO_AVAILABLE = False
else:
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
    os.makedirs(output_dir, exist_ok=True)
    name = generator.__class__.__name__
    logger.info(f"[{name}] Generating batch of size {batch_size}")

    batch = generator.generate_batch(batch_size=batch_size, seed=seed)
    values = torch.from_numpy(batch.values)
    if values.ndim == 2:
        values = values.unsqueeze(-1)

    future_length = sample_future_length(range="gift_eval")
    history_values = values[:, :-future_length, :]
    future_values = values[:, -future_length:, :]

    container = BatchTimeSeriesContainer(
        history_values=history_values,
        future_values=future_values,
        start=batch.start,
        frequency=batch.frequency,
    )

    indices = [sample_idx] if sample_idx is not None else range(batch_size)
    for i in indices:
        filename = (
            f"{prefix}_{name.lower().replace('generatorwrapper', '')}_sample_{i}.png"
        )
        output_file = os.path.join(output_dir, filename)
        title = f"{prefix.capitalize()} {name.replace('GeneratorWrapper', '')} Synthetic Series (Sample {i})"
        plot_from_container(
            container, sample_idx=i, output_file=output_file, show=False, title=title
        )
        logger.info(f"[{name}] Saved plot to {output_file}")


def generator_factory(global_seed: int, total_length: int) -> List:
    generators = [
        KernelGeneratorWrapper(
            KernelGeneratorParams(global_seed=global_seed, length=total_length)
        ),
        GPGeneratorWrapper(
            GPGeneratorParams(global_seed=global_seed, length=total_length)
        ),
        ForecastPFNGeneratorWrapper(
            ForecastPFNGeneratorParams(global_seed=global_seed, length=total_length)
        ),
        SineWaveGeneratorWrapper(
            SineWaveGeneratorParams(global_seed=global_seed, length=total_length)
        ),
        SawToothGeneratorWrapper(
            SawToothGeneratorParams(global_seed=global_seed, length=total_length)
        ),
        StepGeneratorWrapper(
            StepGeneratorParams(global_seed=global_seed, length=total_length)
        ),
        AnomalyGeneratorWrapper(
            AnomalyGeneratorParams(global_seed=global_seed, length=total_length)
        ),
        SpikesGeneratorWrapper(
            SpikesGeneratorParams(global_seed=global_seed, length=total_length)
        ),
        CauKerGeneratorWrapper(
            CauKerGeneratorParams(
                global_seed=global_seed, length=total_length, num_channels=5
            )
        ),
        OrnsteinUhlenbeckProcessGeneratorWrapper(
            OrnsteinUhlenbeckProcessGeneratorParams(
                global_seed=global_seed, length=total_length
            )
        ),
    ]

    if PYO_AVAILABLE:
        generators.extend(
            [
                StochasticRhythmAudioWrapper(
                    StochasticRhythmAudioParams(
                        global_seed=global_seed, length=total_length
                    )
                ),
                FinancialVolatilityAudioWrapper(
                    FinancialVolatilityAudioParams(
                        global_seed=global_seed, length=total_length
                    )
                ),
                MultiScaleFractalAudioWrapper(
                    MultiScaleFractalAudioParams(
                        global_seed=global_seed, length=total_length
                    )
                ),
                NetworkTopologyAudioWrapper(
                    NetworkTopologyAudioParams(
                        global_seed=global_seed, length=total_length
                    )
                ),
            ]
        )
    else:
        logger.warning("Audio generators skipped (pyo not available)")

    return generators


if __name__ == "__main__":
    batch_size = 2
    total_length = 2048
    output_dir = "outputs/plots"
    global_seed = 2025

    logger.info(f"Saving plots to {output_dir}")

    for gen in generator_factory(global_seed, total_length):
        prefix = "multivariate" if getattr(gen.params, "num_channels", 1) > 1 else ""
        visualize_batch_sample(
            gen,
            batch_size=batch_size,
            output_dir=output_dir,
            prefix=prefix,
            seed=global_seed,
        )
