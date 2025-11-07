import argparse
import logging

import torch
from examples.utils import (
    load_model,
    run_inference_and_plot,
)
from huggingface_hub import hf_hub_download
from src.data.containers import BatchTimeSeriesContainer
from src.synthetic_generation.generator_params import SineWaveGeneratorParams
from src.synthetic_generation.sine_waves.sine_wave_generator_wrapper import (
    SineWaveGeneratorWrapper,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    """Main execution function."""
    # CLI
    parser = argparse.ArgumentParser(description="Quick start demo for TimeSeriesModel")
    parser.add_argument(
        "--config",
        default="configs/example.yaml",
        help="Path to model config YAML (default: configs/example.yaml)",
    )
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--total_length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()

    # Configuration
    batch_size = args.batch_size
    total_length = args.total_length
    output_dir = args.output_dir
    seed = args.seed
    config_path = args.config

    logger.info("Downloading model checkpoint from Hugging Face Hub...")
    model_path = hf_hub_download(repo_id="AutoML-org/TempoPFN", filename="models/checkpoint_38M.pth")
    logger.info(f"Checkpoint downloaded to: {model_path}")

    logger.info("=== Time Series Model Demo (Univariate Quantile) ===")

    # 1) Generate synthetic sine wave data
    sine_params = SineWaveGeneratorParams(global_seed=seed, length=total_length)
    sine_generator = SineWaveGeneratorWrapper(sine_params)
    batch = sine_generator.generate_batch(batch_size=batch_size, seed=seed)
    values = torch.from_numpy(batch.values).to(torch.float32)
    if values.ndim == 2:
        values = values.unsqueeze(-1)  # Ensure [B, S, 1] for univariate
    future_length = 256
    history_values = values[:, :-future_length, :]
    future_values = values[:, -future_length:, :]

    # 2) Load the pretrained model (CUDA-only). This demo requires a CUDA GPU.
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this demo. No CUDA device detected.")
    device = torch.device("cuda:0")
    model = load_model(config_path=config_path, model_path=model_path, device=device)

    # 3) Pack tensors into the model's input container
    container = BatchTimeSeriesContainer(
        history_values=history_values.to(device),
        future_values=future_values.to(device),
        start=batch.start,
        frequency=batch.frequency,
    )

    # 4) Run inference (bfloat16 on CUDA) and plot results
    run_inference_and_plot(model=model, container=container, output_dir=output_dir, use_bfloat16=True)

    logger.info("=== Demo completed successfully! ===")


if __name__ == "__main__":
    main()
