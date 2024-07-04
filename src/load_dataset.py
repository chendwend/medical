import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


def load_dataset(cfg, task, testing=False):
    data_transforms = transforms.Compose(
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
    train_dataset = torchvision.datasets.ImageFolder(
        root=str(cfg["folders"][task]) + "/train", transform=data_transforms
    )
    val_dataset = torchvision.datasets.ImageFolder(
        root=str(cfg["folders"][task]) + "/val", transform=data_transforms
    )
    test_dataset = torchvision.datasets.ImageFolder(
        root=str(cfg["folders"][task]) + "/test", transform=data_transforms
    )

    # train_dataset, val_dataset, test_dataset = random_split(
    #     dataset, [cfg.hp.split.train, cfg.hp.split.val, cfg.hp.split.test]
    # )
    if testing:
        indices = np.random.choice(len(train_dataset), size=500, replace=False)
        train_dataset = Subset(train_dataset, indices)

    train_loader = DataLoader(train_dataset, batch_size=cfg.hp.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.hp.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.hp.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
