import argparse
import logging
import os
import warnings
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.optim as optim
import torchmetrics
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import wandb
from src.data.containers import BatchTimeSeriesContainer
from src.data.loaders import SyntheticValidationDataset, create_synthetic_dataset
from src.gift_eval.aggregate_results import aggregate_results
from src.gift_eval.constants import ALL_DATASETS
from src.gift_eval.evaluate import evaluate_in_memory
from src.models.model import TimeSeriesModel
from src.optim.lr_scheduler import WarmupStableDecayScheduler, get_scheduler
from src.plotting.plot_multivariate_timeseries import plot_from_container
from src.utils.utils import (
    generate_descriptive_model_name,
    seed_everything,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress debug messages from external libraries
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def setup_distributed():
    """Initializes the distributed process group."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_distributed():
    """Cleans up the distributed process group safely."""
    try:
        if dist.is_available() and dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception:
                pass
            try:
                dist.destroy_process_group()
            except Exception as e:
                logger.warning(f"Error during destroy_process_group: {e}")
    except Exception:
        pass


def is_main_process():
    return dist.get_rank() == 0


class TrainingPipeline:
    def __init__(self, config: Dict):
        self.config = config
        self.grad_accum_enabled = bool(
            self.config.get("gradient_accumulation_enabled", False)
        )
        self.accumulation_steps = (
            max(1, int(self.config.get("accumulation_steps", 1)))
            if self.grad_accum_enabled
            else 1
        )

        # --- Distributed Setup ---
        self.local_rank = setup_distributed()
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.local_rank}")

        self.initial_epoch = 0
        self.wandb_step_offset = 0
        self._setup()

        if is_main_process():
            logger.info("Loaded config:")
            for key, value in self.config.items():
                logger.info(f"{key}: {value}")

    def _setup(self) -> None:
        seed_everything(self.config["seed"])
        self.config["model_name"] = generate_descriptive_model_name(self.config)

        # Resolve run output directory
        self.run_output_dir = (
            self.config.get("run_output_dir")
            or f"{self.config['model_path']}/{self.config['model_name']}"
        )
        self.config["resolved_run_output_dir"] = self.run_output_dir


        if is_main_process() and self.config.get("wandb"):
            init_kwargs = {
                "name": self.config["model_name"],
                "resume": "allow",  # Allows resuming a run if the ID exists
            }

            # Allow selecting which account/team (entity) to log runs to
            # If not provided, W&B will use the default entity for the API key
            if self.config.get("wandb_entity"):
                init_kwargs["entity"] = self.config.get("wandb_entity")

            # If continuing training, try to load the previous run ID
            if self.config.get("continue_training"):
                if self.config.get("wandb_run_id"):
                    init_kwargs["id"] = self.config["wandb_run_id"]
                    logger.info(
                        f"Attempting to resume wandb run with ID: {self.config['wandb_run_id']}"
                    )

            # Initialize Weights & Biases
            wandb.init(
                project=self.config.get("wandb_project_name", "TimeSeriesForecasting"),
                config=self.config,
                **init_kwargs,
            )

        self.num_training_iterations = self.config.get("num_training_iterations")

        self.model = TimeSeriesModel(**self.config["TimeSeriesModel"]).to(self.device)
        if is_main_process():
            logger.info("=" * 80)
            logger.info(
                f"Initializing model with {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M parameters"
            )
            logger.info("=" * 80)
            logger.info(f"Run output directory: {self.run_output_dir}")
       
        dist.barrier(device_ids=[self.local_rank])
        self._setup_optimizer()
        self._load_checkpoint()
     
        dist.barrier(device_ids=[self.local_rank])
        logger.info(
            f"Distributed training setup: rank {self.rank}, world size {self.world_size}, local rank {self.local_rank}, device {self.device}"
        )
        self.model = DDP(
            self.model, device_ids=[self.local_rank], find_unused_parameters=True
        )
        logger.info(
            f"Distributed Data Parallel model initialized on rank {self.local_rank} with device {self.device}"
        )

        augmentations_config = self.config.get("data_augmentation", {})
        nan_stats_path = augmentations_config.get("nan_stats_path")
        nan_patterns_path = augmentations_config.get("nan_patterns_path")

        chosen_scaler_name = self.config.get("TimeSeriesModel", {}).get("scaler")

        # 1. Create the dataset object with rank-based file sharding for scalability
        self.train_dataset = create_synthetic_dataset(
            base_data_dir=self.config.get("train_data_path"),
            batch_size=self.config.get("batch_size", 128),
            num_batches_per_epoch=self.num_training_iterations,
            generator_proportions=self.config.get("generator_proportions"),
            augmentations=augmentations_config,
            augmentation_probabilities=self.config.get("augmentation_probabilities"),
            global_seed=self.config["seed"] + int(os.environ["LOCAL_RANK"]),
            nan_stats_path=nan_stats_path,
            nan_patterns_path=nan_patterns_path,
            chosen_scaler_name=chosen_scaler_name,
            rank=self.rank,
            world_size=self.world_size,
        )

        # 2. Create the DistributedSampler
        train_sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
        )

        # 3. Create the custom collate function
        def collate_fn(batch):
            # Each item from ComposedDataset is already a complete batch container
            return batch[0]

        # 4. Create the final DataLoader
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=1,  # Each dataset item is a full batch
            sampler=train_sampler,
            num_workers=self.config.get("num_workers", 1),
            pin_memory=True,
            collate_fn=collate_fn,
        )
        print(
            f"Distributed DataLoader created with {len(self.train_loader)} batches and num workers={self.config.get('num_workers', 0)}"
        )

        # Validation loader with per-rank file sharding for scalability
        val_dataset = SyntheticValidationDataset(
            base_data_dir=self.config.get("train_data_path"),
            batch_size=self.config.get("validation_batch_size", 64),
            num_batches=self.config.get("num_validation_batches", 1),
            future_length=512,
            generator_proportions=self.config.get("generator_proportions"),
            device=self.device,
            global_seed=self.config["seed"],
            augmentations=augmentations_config,
            augmentation_probabilities=self.config.get("augmentation_probabilities"),
            chosen_scaler_name=chosen_scaler_name,
            nan_stats_path=nan_stats_path,
            nan_patterns_path=nan_patterns_path,
            rank=self.rank,
            world_size=self.world_size,
        )
        val_sampler = DistributedSampler(val_dataset, shuffle=False)

        self.val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,  # Each item from val_dataset is already a complete batch
            shuffle=False,
            sampler=val_sampler,
            collate_fn=collate_fn,
            num_workers=0,  
        )

        self._setup_metrics()

    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler with enhanced WSD support."""
        optimizer_config = {
            "lr": float(self.config["peak_lr"]),
            "weight_decay": float(self.config.get("weight_decay", 0.01)),
            "betas": (
                float(self.config.get("beta1", 0.9)),
                float(self.config.get("beta2", 0.98)),
            ),
            "eps": float(self.config.get("optimizer_eps", 1e-6)),
        }
        self.optimizer = optim.AdamW(self.model.parameters(), **optimizer_config)

        # Calculate scheduler parameters
        effective_accum_steps = self.accumulation_steps
        total_steps = int(
            self.num_training_iterations // effective_accum_steps // self.world_size
        )

        scheduler_type = self.config.get("lr_scheduler", "warmup_stable_decay")

        if scheduler_type == "warmup_stable_decay":
            # Calculate phase durations
            warmup_ratio = float(
                self.config.get("warmup_ratio", 0.01)
            )  # 1% of training
            stable_ratio = float(
                self.config.get("stable_ratio", 0.85)
            )  # 85% of training

            num_warmup_steps = int(total_steps * warmup_ratio)
            num_stable_steps = int(total_steps * stable_ratio)

            # Use the standalone scheduler class for better control
            self.scheduler = WarmupStableDecayScheduler(
                optimizer=self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_stable_steps=num_stable_steps,
                total_steps=total_steps,
                min_lr_ratio=self.config.get("min_lr_ratio", 0.01),
                decay_type=self.config.get("decay_type", "cosine"),
                verbose=is_main_process(),
            )

            if is_main_process():
                logger.info("WSD Scheduler configured:")
                logger.info(f"  Total steps: {total_steps}")
                logger.info(
                    f"  Warmup steps: {num_warmup_steps} ({warmup_ratio * 100:.1f}%)"
                )
                logger.info(
                    f"  Stable steps: {num_stable_steps} ({stable_ratio * 100:.1f}%)"
                )
                logger.info(
                    f"  Decay steps: {total_steps - num_warmup_steps - num_stable_steps}"
                )
                logger.info(f"  Peak LR: {self.config['peak_lr']}")
                logger.info(
                    f"  Min LR: {self.config['peak_lr'] * float(self.config.get('min_lr_ratio', 0.01))}"
                )

        elif scheduler_type == "cosine_with_warmup":
            num_warmup_steps = int(total_steps * self.config.get("warmup_ratio", 0.01))

            self.scheduler = get_scheduler(
                scheduler_type="cosine_with_warmup",
                optimizer=self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps,
                scheduler_kwargs={
                    "min_lr_ratio": float(self.config.get("min_lr_ratio", 0.01)),
                    "num_cycles": float(self.config.get("num_cycles", 0.5)),
                },
            )

        elif scheduler_type == "cosine_with_restarts":
            num_warmup_steps = int(total_steps * self.config.get("warmup_ratio", 0.01))

            self.scheduler = get_scheduler(
                scheduler_type="cosine_with_restarts",
                optimizer=self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps,
                scheduler_kwargs={
                    "min_lr_ratio": float(self.config.get("min_lr_ratio", 0.01)),
                    "num_cycles": int(self.config.get("num_restart_cycles", 4)),
                },
            )

        elif scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=float(self.config["peak_lr"])
                * float(self.config.get("min_lr_ratio", 0.01)),
            )

        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

        if is_main_process():
            logger.info(f"Optimizer configured with {scheduler_type} scheduler")

    def _setup_metrics(self):
        self.train_metrics = {
            "mape": torchmetrics.MeanAbsolutePercentageError(
                dist_sync_on_step=False, compute_on_cpu=False, sync_on_compute=True
            ).to(self.device),
            "mse": torchmetrics.MeanSquaredError(
                dist_sync_on_step=False, compute_on_cpu=False, sync_on_compute=True
            ).to(self.device),
            "smape": torchmetrics.SymmetricMeanAbsolutePercentageError(
                dist_sync_on_step=False, compute_on_cpu=False, sync_on_compute=True
            ).to(self.device),
        }
        self.val_metrics = {
            "mape": torchmetrics.MeanAbsolutePercentageError(
                dist_sync_on_step=False, compute_on_cpu=False, sync_on_compute=True
            ).to(self.device),
            "mse": torchmetrics.MeanSquaredError(
                dist_sync_on_step=False, compute_on_cpu=False, sync_on_compute=True
            ).to(self.device),
            "smape": torchmetrics.SymmetricMeanAbsolutePercentageError(
                dist_sync_on_step=False, compute_on_cpu=False, sync_on_compute=True
            ).to(self.device),
        }

    def _load_checkpoint(self):
        # Only attempt to load a checkpoint when continuing training and a path is provided
        if not self.config.get("continue_training"):
            return

        checkpoint_path_value = self.config.get("checkpoint_path")
        if not checkpoint_path_value:
            if is_main_process():
                logger.info(
                    "continue_training=True but no checkpoint_path provided; starting from scratch."
                )
            return

        checkpoint_path = Path(checkpoint_path_value)
        if not checkpoint_path.exists():
            if is_main_process():
                logger.warning(
                    f"Checkpoint path does not exist at {checkpoint_path}. Starting from scratch."
                )
            return

        if is_main_process():
            logger.info(f"Loading checkpoint from: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])

    def _save_checkpoint(self, epoch: int):
        dist.barrier()
        if is_main_process():
            model_dir = self.run_output_dir
            os.makedirs(model_dir, exist_ok=True)

            unwrapped_model = self.model.module
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": unwrapped_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "wandb_run_id": self.config.get("wandb_run_id"),
            }

            if hasattr(self.scheduler, "state_dict"):
                checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
            elif hasattr(self.scheduler, "current_step"):
                checkpoint["wsd_scheduler_state"] = self.scheduler.state_dict()

            checkpoint_path = f"{model_dir}/checkpoint.pth"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Checkpoint saved for step {epoch} to {checkpoint_path}")

            config_path = f"{model_dir}/config.yaml"
            with open(config_path, "w") as config_file:
                yaml.dump(self.config, config_file)

    def _inverse_scale(self, model, output: dict) -> torch.Tensor:
        # Use the unwrapped model (module) to access scaler
        return model.module.scaler.inverse_scale(
            output["result"], output["scale_statistics"]
        )

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        self.train_loader.sampler.set_epoch(epoch)

        train_loss, total_loss_sum, total_samples = 0.0, 0.0, 0.0

        pbar = tqdm(
            self.train_loader,
            desc=f"Training (start_step={epoch})",
            disable=not is_main_process(),
        )

        self.optimizer.zero_grad()

        for i, batch in enumerate(pbar):
            batch_size = batch.history_values.size(0)
            batch.to(self.device)

            with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
                output = self.model(batch)
                loss = self.model.module.compute_loss(batch.future_values, output)

                if self.accumulation_steps > 1:
                    loss = loss / self.accumulation_steps

            loss.backward()

            total_loss_sum += loss.item() * batch_size
            total_samples += batch_size

            if ((i + 1) % self.accumulation_steps == 0) or (
                (i + 1) == len(self.train_loader)
            ):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.get("gradient_clip_val", 1.0)
                )

                self.optimizer.step()

                if hasattr(self.scheduler, "step") and callable(self.scheduler.step):
                    if isinstance(self.scheduler, WarmupStableDecayScheduler):
                        self.scheduler.step()  
                    else:
                        self.scheduler.step()  

                self.optimizer.zero_grad()

            if (i + 1) % self.config.get("log_interval", 10) == 0:
                dist.barrier()
                self._validate_epoch(i)

                total_loss_tensor = torch.tensor(
                    [total_loss_sum, total_samples], device=self.device
                )
                dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
                global_loss_sum, global_samples = total_loss_tensor.tolist()

                train_loss = (
                    global_loss_sum / global_samples if global_samples > 0 else 0.0
                )
                if self.accumulation_steps > 1:
                    train_loss *= self.accumulation_steps

                if is_main_process():
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    step_metrics = {
                        "train/step_loss": train_loss,
                        "train/learning_rate": current_lr,
                        "train/lr_schedule_step": i,
                    }

                    if hasattr(self.scheduler, "get_phase"):
                        step_metrics["train/lr_phase"] = self.scheduler.get_phase()
                        step_metrics["train/lr_factor"] = self.scheduler.get_lr_factor(
                            self.scheduler.current_step - 1
                        )

                    if self.config.get("wandb"):
                        wandb.log(step_metrics, step=i)

                    logger.info(
                        f"Step {i} | Training Loss: {train_loss:.4f} | LR: {current_lr:.2e}"
                    )

                total_loss_sum, total_samples = 0.0, 0

            if (i + 1) % self.config.get("save_every", 10) == 0:
                self._save_checkpoint(i)

        return train_loss

    def _validate_epoch(self, epoch: int) -> float:
        self.model.eval()

        for metric in self.val_metrics.values():
            metric.reset()

        first_batch_for_plotting = None

        total_loss_sum, total_samples = 0.0, 0
        with torch.no_grad():
            self.val_loader.sampler.set_epoch(epoch)
            for batch_idx, batch in enumerate(self.val_loader):
                if is_main_process() and batch_idx == 0:
                    first_batch_for_plotting = batch.to(torch.device("cpu"))

                batch = batch.to(self.device)
                batch_size = batch.history_values.size(0)

                with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
                    output = self.model.module(batch)  # Use unwrapped model
                    loss = self.model.module.compute_loss(batch.future_values, output)

                inv_scaled_output = self._inverse_scale(self.model, output)
                total_loss_sum += loss.item() * batch_size
                total_samples += batch_size

                self._update_metrics(
                    self.val_metrics,
                    inv_scaled_output,
                    batch.future_values,
                    distributed=False,
                )

        total_stats = torch.tensor([total_loss_sum, total_samples], device=self.device)
        dist.all_reduce(total_stats, op=dist.ReduceOp.SUM)
        global_loss_sum, global_samples = total_stats.tolist()
        avg_val_loss = global_loss_sum / global_samples if global_samples > 0 else 0.0

        val_computed_metrics = {
            name: metric.compute() for name, metric in self.val_metrics.items()
        }

        if is_main_process():
            log_metrics = {"val/loss": avg_val_loss}
            log_metrics.update(
                {
                    f"val/{name}": value.item()
                    for name, value in val_computed_metrics.items()
                }
            )

            if self.config.get("wandb"):
                wandb.log(log_metrics, step=epoch + self.wandb_step_offset)

            logger.info(
                f"Epoch {epoch} | Validation Loss: {avg_val_loss:.4f} | Validation MAPE: {val_computed_metrics.get('mape', -1).item():.4f}"
            )

            if first_batch_for_plotting is not None:
                self._plot_validation_examples(
                    epoch, first_batch_for_plotting, plot_all=True
                )

        # Ensure all ranks finish validation before returning to training
        dist.barrier()
        return avg_val_loss

    def _update_metrics(
        self,
        metrics: Dict,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        distributed: bool = True,
    ):
        """
        Gathers tensors if in distributed mode and updates the metric objects.
        """
        if distributed and dist.is_initialized():
            world_size = dist.get_world_size()
            predictions_list = [
                torch.zeros_like(predictions) for _ in range(world_size)
            ]
            targets_list = [torch.zeros_like(targets) for _ in range(world_size)]

            dist.all_gather(predictions_list, predictions)
            dist.all_gather(targets_list, targets)

            predictions_gathered = torch.cat(predictions_list, dim=0)
            targets_gathered = torch.cat(targets_list, dim=0)
        else:
            predictions_gathered = predictions
            targets_gathered = targets

        unwrapped_model = self.model.module

        if unwrapped_model.loss_type == "quantile":
            try:
                median_idx = unwrapped_model.quantiles.index(0.5)
                predictions_gathered = predictions_gathered[..., median_idx]
            except (ValueError, AttributeError):
                if is_main_process():
                    logger.warning(
                        "Median (0.5) quantile not found for metric calculation. Skipping."
                    )
                return  # Exit if we can't get a point forecast

        if predictions_gathered.dim() == 3:
            b, p, c = predictions_gathered.shape
            predictions_flat = predictions_gathered.permute(0, 2, 1).reshape(b * c, p)
            targets_flat = targets_gathered.permute(0, 2, 1).reshape(b * c, p)

            for metric in metrics.values():
                metric.update(predictions_flat, targets_flat)

    def _plot_validation_examples(
        self,
        epoch: int,
        plot_batch: BatchTimeSeriesContainer,
        plot_indices: List[int] = [0, 1, 2, 3, 4],
        plot_all: bool = False,
    ) -> None:
        """
        Plots validation examples from a given batch and logs them to WandB.
        This method should only be called from the main process.
        """
        if (not self.config.get("wandb")) or (
            not self.config.get("wandb_plots", False)
        ):
            return

        model = self.model.module

        with torch.inference_mode():
            plot_batch.to(self.device)

            with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
                output = model(plot_batch)

            inv_scaled_output = self._inverse_scale(self.model, output)
            pred_future = inv_scaled_output.cpu().numpy()

            batch_size = plot_batch.history_values.size(0)
            if plot_all:
                indices_to_plot = list(range(batch_size))
            else:
                indices_to_plot = [i for i in plot_indices if i < batch_size]

            for i in indices_to_plot:
                fig = plot_from_container(
                    batch=plot_batch,
                    sample_idx=i,
                    predicted_values=pred_future,
                    model_quantiles=model.quantiles
                    if model.loss_type == "quantile"
                    else None,
                    title=f"Epoch {epoch} - Val Sample {i}",
                    output_file=None,
                    show=False,
                )

                wandb.log(
                    {f"val_plots/sample_{i}": wandb.Image(fig)},
                    step=epoch + self.wandb_step_offset,
                )
                plt.close(fig)

    def train(self) -> None:
        if is_main_process():
            per_rank_iterations = len(self.train_loader)
            optimizer_steps_per_rank = (
                per_rank_iterations + self.accumulation_steps - 1
            ) // self.accumulation_steps
            logger.info(
                f"Starting training: configured_iterations={self.num_training_iterations}, "
                f"world_size={self.world_size}, per_rank_iterations={per_rank_iterations}, "
                f"accumulation_steps={self.accumulation_steps}, "
                f"optimizer_steps_per_rank={optimizer_steps_per_rank}"
            )

        self._train_epoch(self.initial_epoch)

        dist.barrier()

        if not is_main_process():
            try:
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
            except Exception:
                pass
            cleanup_distributed()
            return

        cleanup_distributed()

        gift_eval_config = self.config.get("gift_eval")
        if gift_eval_config.get("evaluate_on_gift_eval"):
            output_dir = f"{self.run_output_dir}/gift_eval_results"

            evaluate_in_memory(
                model=self.model.module if isinstance(self.model, DDP) else self.model,
                config=self.config,
                datasets=ALL_DATASETS,
                terms=["short", "medium", "long"],
                dataset_storage_path=gift_eval_config.get("dataset_storage_path"),
                batch_size=self.config.get("batch_size"),
                max_context_length=gift_eval_config.get("max_context_length"),
                output_dir=output_dir,
                create_plots=gift_eval_config.get("create_plots"),
                max_plots=gift_eval_config.get("max_plots"),
            )

            aggregate_results(
                result_root_dir=output_dir,
            )

        if self.config.get("wandb"):
            logger.info("TRAINING COMPLETED SUCCESSFULLY!")
            wandb.finish()

        try:
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", default="./configs/train.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--run_output_dir",
        default=None,
        help=(
            "Optional output directory to store checkpoints and artifacts. "
            "If provided, overrides model_path/model_name for saving."
        ),
    )
    args = parser.parse_args()
    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)

    # Allow CLI to override output directory for artifacts/logical run folder
    if getattr(args, "run_output_dir", None):
        config["run_output_dir"] = args.run_output_dir

    try:
        pipeline = TrainingPipeline(config)
        pipeline.train()
    finally:
        # Protect final CUDA ops to avoid raising if device already torn down
        try:
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        except Exception:
            pass