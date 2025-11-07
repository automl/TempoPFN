# src/utils/lr_scheduler.py

import math
from enum import Enum
from functools import partial

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class SchedulerType(Enum):
    """Enumeration of available learning rate schedulers."""

    COSINE = "cosine"
    COSINE_WITH_WARMUP = "cosine_with_warmup"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    WARMUP_STABLE_DECAY = "warmup_stable_decay"
    POLYNOMIAL_WITH_WARMUP = "polynomial_with_warmup"
    LINEAR_WITH_WARMUP = "linear_with_warmup"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
    INVERSE_SQRT = "inverse_sqrt"


def _get_warmup_stable_decay_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_stable_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.001,
    decay_type: str = "cosine",
):
    """
    Learning rate lambda function for Warmup-Stable-Decay (WSD) schedule.

    This scheduler implements three phases:
    1. Warmup: Linear increase from 0 to peak learning rate
    2. Stable: Constant learning rate for majority of training
    3. Decay: Gradual decrease using cosine or linear decay

    Args:
        current_step: Current training step
        num_warmup_steps: Number of warmup steps
        num_stable_steps: Number of stable learning rate steps
        num_training_steps: Total number of training steps
        min_lr_ratio: Minimum learning rate as ratio of peak learning rate
        decay_type: Type of decay schedule ("cosine" or "linear")
    """
    if current_step < num_warmup_steps:
        # Warmup phase: linear increase
        return float(current_step) / float(max(1, num_warmup_steps))

    elif current_step < num_warmup_steps + num_stable_steps:
        # Stable phase: constant learning rate
        return 1.0

    else:
        # Decay phase
        decay_steps = num_training_steps - num_warmup_steps - num_stable_steps
        if decay_steps <= 0:
            return max(min_lr_ratio, 1.0)

        progress = (current_step - num_warmup_steps - num_stable_steps) / decay_steps
        progress = min(progress, 1.0)  # Clamp to [0, 1]

        if decay_type == "cosine":
            # Cosine decay
            decay_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(min_lr_ratio, decay_factor)
        elif decay_type == "linear":
            # Linear decay
            decay_factor = 1.0 - progress
            return max(min_lr_ratio, decay_factor)
        else:
            raise ValueError(f"Unknown decay_type: {decay_type}")


def get_warmup_stable_decay_schedule(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_stable_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.01,
    decay_type: str = "cosine",
    last_epoch: int = -1,
):
    """
    Create a Warmup-Stable-Decay learning rate schedule.

    This scheduler is particularly well-suited for foundation model training as it:
    - Provides stable learning during the majority of training
    - Doesn't require pre-committing to exact training duration
    - Allows for extended training without aggressive decay

    Args:
        optimizer: The optimizer for which to schedule the learning rate
        num_warmup_steps: Number of steps for warmup phase
        num_stable_steps: Number of steps for stable learning rate phase
        num_training_steps: Total number of training steps
        min_lr_ratio: Minimum learning rate as fraction of peak learning rate
        decay_type: Type of decay ("cosine" or "linear")
        last_epoch: The index of the last epoch when resuming training

    Returns:
        torch.optim.lr_scheduler.LambdaLR with the WSD schedule
    """
    lr_lambda = partial(
        _get_warmup_stable_decay_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_stable_steps=num_stable_steps,
        num_training_steps=num_training_steps,
        min_lr_ratio=min_lr_ratio,
        decay_type=decay_type,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    min_lr_ratio: float = 0.0,
):
    """Enhanced cosine schedule with configurable minimum learning rate."""
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
    return max(min_lr_ratio, cosine_factor)


def get_enhanced_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    min_lr_ratio: float = 0.01,
    last_epoch: int = -1,
):
    """
    Enhanced cosine schedule with warmup and configurable minimum learning rate.

    Args:
        optimizer: The optimizer for which to schedule the learning rate
        num_warmup_steps: Number of steps for warmup phase
        num_training_steps: Total number of training steps
        num_cycles: Number of cosine cycles (0.5 = half cosine)
        min_lr_ratio: Minimum learning rate as fraction of peak learning rate
        last_epoch: The index of the last epoch when resuming training
    """
    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        min_lr_ratio=min_lr_ratio,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def _get_cosine_with_restarts_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int = 1,
    min_lr_ratio: float = 0.0,
):
    """Cosine schedule with hard restarts and configurable minimum learning rate."""
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    if progress >= 1.0:
        return min_lr_ratio

    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0)))
    return max(min_lr_ratio, cosine_factor)


def get_cosine_with_restarts_schedule(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int = 4,
    min_lr_ratio: float = 0.01,
    last_epoch: int = -1,
):
    """
    Cosine schedule with hard restarts.

    Args:
        optimizer: The optimizer for which to schedule the learning rate
        num_warmup_steps: Number of steps for warmup phase
        num_training_steps: Total number of training steps
        num_cycles: Number of restart cycles
        min_lr_ratio: Minimum learning rate as fraction of peak learning rate
        last_epoch: The index of the last epoch when resuming training
    """
    lr_lambda = partial(
        _get_cosine_with_restarts_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        min_lr_ratio=min_lr_ratio,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


# Scheduler registry for easy lookup
SCHEDULER_REGISTRY = {
    SchedulerType.WARMUP_STABLE_DECAY: get_warmup_stable_decay_schedule,
    SchedulerType.COSINE_WITH_WARMUP: get_enhanced_cosine_schedule_with_warmup,
    SchedulerType.COSINE_WITH_RESTARTS: get_cosine_with_restarts_schedule,
}


def get_scheduler(
    scheduler_type: str | SchedulerType,
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    scheduler_kwargs: dict | None = None,
):
    """
    Unified interface to create learning rate schedulers.

    Args:
        scheduler_type: Type of scheduler to create
        optimizer: The optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total training steps
        scheduler_kwargs: Additional scheduler-specific parameters

    Returns:
        Configured learning rate scheduler
    """
    if isinstance(scheduler_type, str):
        scheduler_type = SchedulerType(scheduler_type)

    if scheduler_kwargs is None:
        scheduler_kwargs = {}

    if scheduler_type not in SCHEDULER_REGISTRY:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    scheduler_func = SCHEDULER_REGISTRY[scheduler_type]
    return scheduler_func(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        **scheduler_kwargs,
    )


class WarmupStableDecayScheduler:
    """
    Alternative implementation as a standalone scheduler class.

    This provides more flexibility and better state management for
    complex training scenarios with checkpointing.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_stable_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.01,
        decay_type: str = "cosine",
        verbose: bool = False,
    ):
        self.optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.num_stable_steps = num_stable_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.decay_type = decay_type
        self.verbose = verbose

        # Store initial learning rates
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.current_step = 0

    def get_lr_factor(self, step: int) -> float:
        """Calculate the learning rate multiplication factor for given step."""
        if step < self.num_warmup_steps:
            # Warmup phase
            return step / max(1, self.num_warmup_steps)
        elif step < self.num_warmup_steps + self.num_stable_steps:
            # Stable phase
            return 1.0
        else:
            # Decay phase
            decay_steps = self.total_steps - self.num_warmup_steps - self.num_stable_steps
            if decay_steps <= 0:
                return max(self.min_lr_ratio, 1.0)

            progress = (step - self.num_warmup_steps - self.num_stable_steps) / decay_steps
            progress = min(progress, 1.0)

            if self.decay_type == "cosine":
                decay_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            elif self.decay_type == "linear":
                decay_factor = 1.0 - progress
            else:
                raise ValueError(f"Unknown decay_type: {self.decay_type}")

            return max(self.min_lr_ratio, decay_factor)

    def step(self):
        """Update learning rates for all parameter groups."""
        lr_factor = self.get_lr_factor(self.current_step)

        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            param_group["lr"] = base_lr * lr_factor

        if self.verbose and self.current_step % 1000 == 0:
            phase = self.get_phase()
            print(f"Step {self.current_step}: LR factor = {lr_factor:.6f}, Phase = {phase}")

        self.current_step += 1

    def get_phase(self) -> str:
        """Get current training phase."""
        if self.current_step < self.num_warmup_steps:
            return "warmup"
        elif self.current_step < self.num_warmup_steps + self.num_stable_steps:
            return "stable"
        else:
            return "decay"

    def state_dict(self) -> dict:
        """Return scheduler state for checkpointing."""
        return {
            "current_step": self.current_step,
            "base_lrs": self.base_lrs,
        }

    def load_state_dict(self, state_dict: dict):
        """Load scheduler state from checkpoint."""
        self.current_step = state_dict["current_step"]
        self.base_lrs = state_dict["base_lrs"]
