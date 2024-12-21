from datetime import datetime
from pathlib import Path
from typing import Tuple

import torch
import torch.distributed as dist
from omegaconf import DictConfig
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

# import wandb
from metrics import metrics_calc_func
# from metrics.metrics import Metrics
from utils import ddp_utils, general_utils, torch_utils

# from tqdm import tqdm



class Trainer():
     def __init__(
        self,
        task: str,
        model: torch.nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        loss_func, 
        class_labels: list, 
        hp_dict: dict,
        patience: int,
        world_size: int,
        master_process: bool,
        device: int,
        testing: bool,
    ) -> None:
         NotImplemented 