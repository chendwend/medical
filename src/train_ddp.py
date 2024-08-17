import os
import pickle

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig

import wandb
from dataloaders import get_dataloaders
from model import CustomResNet
from Trainer import Trainer


def broadcast_object(obj, master_process):

    device = f'cuda:{int(os.environ["LOCAL_RANK"])}'
    if master_process:
        obj_bytes = pickle.dumps(obj)
        obj_size = torch.tensor(len(obj_bytes), dtype=torch.long, device=device)
    else:
        obj_size = torch.tensor(0, dtype=torch.long, device=device)

    dist.broadcast(obj_size, 0)
    obj_bytes = bytearray(obj_size.item())

    if master_process:
        # Only the source fills the byte buffer
        obj_bytes[:] = pickle.dumps(obj)

    obj_tensor = torch.ByteTensor(obj_bytes).to(device)
    dist.broadcast(obj_tensor, 0)

    # Deserialize the byte stream back into the Python object
    obj = pickle.loads(obj_tensor.cpu().numpy().tobytes())
    torch.distributed.barrier() 
    return obj

def setup(seed):
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    torch.manual_seed(seed) 
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    master_process = rank == 0
    torch.cuda.set_device(rank)
    return rank, world_size, master_process

def cleanup():
    dist.destroy_process_group()
    torch.cuda.empty_cache()

def train(cfg):
    try:
        rank, world_size, master_process = setup(cfg["seed"])
        if rank == 0:
            wandb.init()
            config = wandb.config
            hp_dict = {k: v for k, v in dict(config).items()}
            param_string = "_".join([f"{k}={v}" for k, v in hp_dict.items()])
            wandb.run.name = cfg.task + "_" + cfg.model + "_" + param_string
        else:
            hp_dict = None
        hp_dict = broadcast_object(hp_dict, rank==0)

        train_loader, val_loader, test_loader = get_dataloaders(cfg["folders"][cfg.task], 
                                                                cfg.preprocessing.image_size, 
                                                                cfg.preprocessing.norm, 
                                                                hp_dict["batch_size"], 
                                                                world_size, 
                                                                rank, 
                                                                cfg.testing)
        model = CustomResNet(num_classes=cfg["classes_per_task"][cfg.task], 
                             model_name=cfg.model, 
                             fc_layer=hp_dict["fc_layer"])
        loss = torch.nn.CrossEntropyLoss(label_smoothing=hp_dict["label_smoothing"])

        trainer = Trainer(cfg.task, 
                            model, 
                            train_loader, 
                            val_loader, 
                            loss,
                            cfg["class_names"][cfg.task],
                            hp_dict,
                            cfg["early_stopping_patience"],
                            world_size=world_size,
                            master_process=master_process,
                            testing=cfg.testing,
                            )
        exec_time = trainer.train_model()


    finally:
        cleanup()


def test_num_workers(cfg:DictConfig):
    try:
        rank, world_size, master_process = setup(cfg["seed"])

        hp_dict = cfg.hp
        hp_dict.batch_size *= world_size
        hp_dict = broadcast_object(hp_dict, rank==0)

        train_loader, val_loader, test_loader = get_dataloaders(cfg["folders"][cfg.task], 
                                                                cfg.preprocessing.image_size, 
                                                                cfg.preprocessing.norm, 
                                                                hp_dict["batch_size"], 
                                                                world_size, 
                                                                rank, 
                                                                num_workers,
                                                                False)
        model = CustomResNet(num_classes=cfg["classes_per_task"][cfg.task], 
                             model_name=cfg.model, 
                             fc_layer=hp_dict["fc_layer"])
        loss = torch.nn.CrossEntropyLoss(label_smoothing=hp_dict["label_smoothing"])

        trainer = Trainer(cfg.task, 
                            model, 
                            train_loader, 
                            val_loader, 
                            loss,
                            cfg["class_names"][cfg.task],
                            hp_dict,
                            cfg["early_stopping_patience"],
                            world_size=world_size,
                            master_process=master_process,
                            testing=True,
                            )
        exec_time = trainer.train_model()

    finally:
        cleanup()

    return exec_time

def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--model" , help="model type",  choices=["resnet50", "resnet101", "resnet152"])
    parser.add_argument("-t", "--task", help="type of task", choices=["pathology", "birads", "mass_shape"])
    parser.add_argument("-w", "--workers", help="test num_workes", action="store_true")

    return parser.parse_args()

if __name__ == "__main__":


    args = parse_args()

    overrides = [
        f"model={args.model}",
        f"task={args.task}",
        f"workers={args.workers}"
    ]
    with hydra.initialize(version_base="1.3", config_path="../conf"):
        cfg = hydra.compose(config_name="config", overrides=overrides)
        if args.workers:
            from multiprocessing import cpu_count

            numWorkers_perf = []

            for num_workers in range(0, cpu_count()+1):
                exec_time = test_num_workers(cfg)
                print(f"{num_workers} workers: {exec_time}")
                numWorkers_perf.append((num_workers, exec_time))

            best_workers = numWorkers_perf.min(lambda f: f[1])
            print(f"best num_workers: {best_workers[0]}")
        else:
            train(cfg)