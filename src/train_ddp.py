import hydra
import torch
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb
from dataloaders import get_dataloaders
from model import CustomResNet
from Trainer import Trainer
from utils import ddp_utils, general_utils


def train(cfg):
    try:
        global_rank, local_rank, world_size, master_process, device = ddp_utils.ddp_setup()
        general_utils.seed_everything(cfg.seed, global_rank)
        if global_rank == 0:
            wandb.init()
            config = wandb.config
            hp_dict = {k: v for k, v in dict(config).items()}
            param_string = "_".join([f"{k}={v}" for k, v in hp_dict.items()])
            wandb.run.name = cfg.task + "_" + cfg.model + "_" + param_string
        else:
            # if not master_process, then no wandb initialization, thus no access to wandb hyperparameters dict
            hp_dict = None
            
        hp_dict = ddp_utils.broadcast_object(hp_dict, global_rank==0)

        print(f"GPU {local_rank} - loading dataset") 
        train_loader, val_loader, test_loader = get_dataloaders(cfg["folders"][cfg.task], 
                                                                cfg.preprocessing.image_size, 
                                                                cfg.preprocessing.norm, 
                                                                hp_dict["batch_size"], 
                                                                world_size, 
                                                                local_rank,
                                                                cfg.seed, 
                                                                cfg.testing)
        model = CustomResNet(num_classes=cfg["classes_per_task"][cfg.task], 
                             model_name=cfg.model, 
                             fc_layer=hp_dict["fc_layer"]).to(local_rank)
        model = DDP(model, device_ids=[local_rank])

        loss = torch.nn.CrossEntropyLoss(label_smoothing=hp_dict["label_smoothing"]).to(device)

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
        ddp_utils.ddp_cleanup()
        if exec_time:
            print(f"Execution time: {exec_time}")


def test_num_workers(cfg:DictConfig):
    try:
        rank, world_size, master_process = ddp_utils.ddp_setup(cfg["seed"])

        hp_dict = cfg.hp
        hp_dict.batch_size *= world_size
        hp_dict = ddp_utils.broadcast_object(hp_dict, rank==0)

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
        ddp_utils.cleanup()

    return exec_time

def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--model" , help="model type",  choices=["resnet50", "resnet101", "resnet152", "resnetv2_50x1_bit"])
    parser.add_argument("-t", "--task", help="type of task", choices=["pathology", "birads", "mass_shape", "breast_density"])
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