import json
import os

import hydra
import torch
from dotenv import load_dotenv
from torch.distributed import (broadcast, destroy_process_group,
                               init_process_group)

import wandb
from src.dataloaders import get_dataloaders
from src.model import CustomResNet
from src.Trainer import Trainer


def setup_wandb():
    load_dotenv()
    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        raise ValueError("WANDB_API_KEY is not set in the environment variables.")
    
    sweep_config = {
        'method': 'random',
        'metric': {'goal': 'minimize', 'name': 'val/loss'},
        'parameters': {
            'epochs': {
                'values': [4]
                # 'values': [30, 50]
            },
            'lr': {
                'distribution': 'log_uniform_values',
                'min': 1e-3,
                'max': 1
            },
            'weight_decay': {
                'values': [1e-3, 1e-4, 1e-5, 0]
            },
            'batch_size':{
                # 'values': [32, 64, 128]
                'values': [32]
            },
            'label_smoothing':{
                'values':  [0.25, 0.35]
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project='medical')
    return sweep_id

def setup_ddp(seed:int) -> bool:
    """ 
    Performs all neccesary setup

    Args:
        seed (int): seed for setting randomization
        hp (dict): dictionary of hyperparameters

    Returns:
        bool: Indication whether ddp training.
    """

    init_process_group(backend="nccl")
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
 
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        master_process = ddp_local_rank == 0
        print("Initiating distributed training...")
    else:
        raise("ddp failed to initiate.")
        master_process = True
        ddp_world_size = 1

    torch.manual_seed(seed+ddp)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Computation device: {DEVICE}\n")
    
    return ddp, master_process, ddp_world_size

def cleanup(ddp):
    if ddp:
        destroy_process_group()
    torch.cuda.empty_cache()


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg):

    sweep_id = setup_wandb()
    print(f"Initiating with sweep counts = {cfg.sweep_counts}.")
    wandb.agent(sweep_id, function=lambda: ddp_trainer(cfg), count=cfg.sweep_counts)
     

def ddp_trainer(cfg):
    ddp, master_process, world_size = setup_ddp(cfg.seed)
    if master_process:
        wandb.init(project="medical")
        config = wandb.config
        hp_dict = {k: v for k, v in dict(config).items()}
        print("---------------config:------------")       
        print(config)
        print("-----------------------------------")
    else:
        hp_dict = None

    hp_dict = broadcast_config(hp_dict, master_process)

    train(cfg, hp_dict, world_size, master_process, ddp)

    cleanup(ddp)

def train(cfg, hp_dict, world_size, master_process, ddp):

    train_loader, val_loader, test_loader = get_dataloaders(cfg["folders"][cfg.task], cfg.preprocessing.image_size, cfg.preprocessing.norm, hp_dict["batch_size"], ddp, cfg.testing)
    model = CustomResNet(num_classes=cfg["classes_per_task"][cfg.task], model_name="resnet50")
    loss = torch.nn.CrossEntropyLoss(label_smoothing=hp_dict["label_smoothing"])

    trainer = Trainer(cfg.task, 
                        model, 
                        train_loader, 
                        val_loader, 
                        loss,
                        cfg["class_names"][cfg.task],
                        hp_dict,
                        world_size=world_size,
                        master_process=master_process,
                        testing=cfg.testing,
                        )
    trainer.train_model()

def broadcast_config(config_dict, master_process):
    device = f'cuda:{int(os.environ["LOCAL_RANK"])}'
    if master_process:
        config_json = json.dumps(config_dict)
        config_bytes = torch.tensor(bytearray(config_json.encode('utf-8')), device=device, dtype=torch.uint8)
    else:
        config_bytes = torch.empty(0, device=device, dtype=torch.uint8)

    # Broadcast the config bytes from the master process to all processes
    broadcast(config_bytes, src=0)

    if not master_process:
        # Deserialize the config bytes back into a dictionary
        config_json = bytes(config_bytes.tolist()).decode('utf-8')
        config_dict = json.loads(config_json.cpu())

    return config_dict

if __name__ == "__main__":
    os.system('clear')
    main()
