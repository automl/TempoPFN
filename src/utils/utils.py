import os
import random
from datetime import datetime

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def generate_descriptive_model_name(config):
    return (
        f"{config['model_name']}_"
        f"BATCH{config['batch_size']}_"
        f"ITER{config['num_training_iterations']}_"
        f"ACCUM_{config['gradient_accumulation_enabled']}_"
        f"ACC_STEPS{config['accumulation_steps']}_"
        f"Emb{config['TimeSeriesModel']['embed_size']}_"
        f"L{config['TimeSeriesModel']['num_encoder_layers']}_"
        f"H{config['TimeSeriesModel']['encoder_config']['num_householder']}_"
        f"LR_SCHEDULER_{config['lr_scheduler']}_"
        f"PEAK_LR{config['peak_lr']}_"
        f"{datetime.now().strftime('_%Y%m%d_%H%M%S')}"
    )
