from os import environ

import hydra
import torch
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb
from dataloaders import get_dataloaders
from model import CustomResNet

# from Trainer import Trainer

# from utils import ddp_utils, general_utils


def main():

    print("Hello")
    print("goodbye")




if __name__ == "__main__":
    main()