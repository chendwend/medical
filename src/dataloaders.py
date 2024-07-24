import os

import numpy as np
import timm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder

# data_config = timm.data.resolve_model_data_config(model)
# transforms = timm.data.create_transform(**data_config, is_training=False)

def prepare_dataloader(dataset: ImageFolder, batch_size: int, is_distributed) -> DataLoader:
    sampler = DistributedSampler(dataset) if is_distributed else None
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=sampler is None, 
        pin_memory=True, 
        sampler=sampler)
    
    return dataloader

def get_dataloaders(cfg, task, is_distributed, testing=False):
    train_transforms = transforms.Compose(
        [
            transforms.Resize(
                (cfg.preprocessing.image_size[0], cfg.preprocessing.image_size[1])
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize(
                (cfg.preprocessing.norm.mean),
                (cfg.preprocessing.norm.std),
            ),
        ]
    )
    val_transforms = transforms.Compose(
        [
            transforms.Resize(
                (cfg.preprocessing.image_size[0], cfg.preprocessing.image_size[1])
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                (cfg.preprocessing.norm.mean),
                (cfg.preprocessing.norm.std),
            ),
        ]
    )
    # Create the dataset
    train_dataset = ImageFolder(
        root=str(cfg["folders"][task]) + "/train", transform=train_transforms
    )
    val_dataset = ImageFolder(
        root=str(cfg["folders"][task]) + "/val", transform=val_transforms
    )
    test_dataset = ImageFolder(
        root=str(cfg["folders"][task]) + "/test", transform=val_transforms
    )

    if testing:
        train_indices = np.random.choice(len(train_dataset), size=int(0.1*len(train_dataset)), replace=False)
        val_indices = np.random.choice(len(val_dataset), size=int(0.1*len(val_dataset)), replace=False)
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)

    if int(os.environ['LOCAL_RANK']) == 0:
        print(f"Train dataset has {len(train_dataset)} images")
        print(f"Val dataset has {len(val_dataset)} images")

    train_loader = prepare_dataloader(train_dataset, is_distributed, cfg.hp.batch_size)
    val_loader   = prepare_dataloader(val_dataset, is_distributed, cfg.hp.batch_size)
    test_loader  = prepare_dataloader(test_dataset, is_distributed, cfg.hp.batch_size)


    return train_loader, val_loader, test_loader
