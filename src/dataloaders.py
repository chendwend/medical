import random
from pathlib import Path
from typing import Union

import numpy as np
import torch
# import timm
import torchvision.transforms as transforms
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder

# data_config = timm.data.resolve_model_data_config(model)
# transforms = timm.data.create_transform(**data_config, is_training=False)

def prepare_dataloader(dataset: ImageFolder, batch_size: int, world_size:int, rank:int, num_workers:int, seed:int, shuffle:bool) -> DataLoader:
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=seed)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=sampler is None, 
        pin_memory=True, 
        sampler=sampler,
        num_workers=num_workers,
        worker_init_fn = worker_init_fn,
        persistent_workers=True)
    # ! fix me
    #FIXME:
    return dataloader

def get_dataloaders(task_parent_dir:Union[str, Path], 
                    image_size:tuple[int, int], 
                    norm_params:dict, 
                    batch_size:int, 
                    world_size:int, 
                    rank:int, 
                    num_workers:int,
                    seed:int,
                    testing=False) -> tuple[DataLoader, DataLoader]:
    
    train_transforms = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            # transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize(
                (norm_params["mean"]),
                (norm_params["std"]),
            ),
        ]
    )
    val_transforms = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                (norm_params["mean"]),
                (norm_params["std"]),
            ),
        ]
    )
    # Create the dataset
    train_dataset = ImageFolder(
        root=str(task_parent_dir) + "/train", transform=train_transforms
    )
    val_dataset = ImageFolder(
        root=str(task_parent_dir) + "/val", transform=val_transforms
    )
    test_dataset = ImageFolder(
        root=str(task_parent_dir) + "/test", transform=val_transforms
    )

    if testing:
        train_indices = np.random.choice(len(train_dataset), size=int(0.1*len(train_dataset)), replace=False)
        val_indices = np.random.choice(len(val_dataset), size=int(0.1*len(val_dataset)), replace=False)
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)

    if rank == 0:
        print(f"Train dataset has {len(train_dataset)} images")
        print(f"Val dataset has {len(val_dataset)} images")

    train_loader = prepare_dataloader(train_dataset, batch_size, world_size, rank, num_workers, seed, shuffle=True)
    val_loader   = prepare_dataloader(val_dataset, batch_size, world_size, rank, num_workers, seed, shuffle=False)
    test_loader  = prepare_dataloader(test_dataset, batch_size, world_size, rank, num_workers, seed, shuffle=False)


    return train_loader, val_loader, test_loader


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main(cfg:DictConfig):
        train_loader, val_loader, test_loader = get_dataloaders(
             cfg.folders.pathology,
             cfg.preprocessing.image_size,
             cfg.preprocessing.norm,
             batch_size=32,
             world_size=1,
             rank=0,
             num_workers=2
        )


if __name__ == "__main__":
    import hydra

    with hydra.initialize(version_base="1.3", config_path="../conf"):
        cfg = hydra.compose(config_name="config")


        main(cfg)
