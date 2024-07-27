import json
import os
import pickle
import hydra
import torch
from dotenv import load_dotenv
from torch.distributed import (broadcast, destroy_process_group,
                               init_process_group, is_initialized)

import wandb
from src.dataloaders import get_dataloaders
from src.model import CustomResNet
from src.Trainer import Trainer
import subprocess


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
                'values': [2]
                # 'values': [30]
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

# def setup_ddp(seed:int) -> bool:
#     """ 
#     Performs all neccesary ddp setup

#     """
#     if not is_initialized():
#         init_process_group(backend="nccl")

#         ddp_local_rank = int(os.environ['LOCAL_RANK'])
#         device = f'cuda:{ddp_local_rank}'
#         torch.cuda.set_device(device)
    
#         ddp_world_size = int(os.environ['WORLD_SIZE'])
#         master_process = ddp_local_rank == 0
#         if master_process:
#             print("Initiating distributed training...")
#     else:
#         ddp_local_rank = int(os.environ['LOCAL_RANK'])
#         ddp_world_size = int(os.environ['WORLD_SIZE'])
#         master_process = ddp_local_rank == 0

#     torch.manual_seed(seed)
            
#     return master_process, ddp_world_size

# def cleanup():
#     destroy_process_group()
#     torch.cuda.empty_cache()


def run_training():
    wandb.init()
    print("here")
    print(os.getcwd())
    subprocess.run(["torchrun", "--nproc_per_node=2", "--nnodes=1", "src/train_ddp.py"])



@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg):
    # print(os.getenv("LOCAL_RANK", "0"))
    sweep_id = setup_wandb()
    print(f"Initiating with sweep counts = {cfg.sweep_counts}.")
    print(os.getcwd())
    wandb.agent(sweep_id, function=run_training, count=cfg.sweep_counts, project="medical")

     

# def ddp_trainer(cfg, ddp_world_size, master_process):
#     try:
#         # master_process, world_size = setup_ddp(cfg.seed)
#         device = f'cuda:{int(os.environ["LOCAL_RANK"])}'
#         if master_process:
#             wandb.init(project="medical")
#             config = wandb.config
#             hp_dict = {k: v for k, v in dict(config).items()}
#             print("---------------config:------------")       
#             print(config)
#             print("-----------------------------------")
#         else:
#             hp_dict = "empty"
#         print(f"\n1: {device}, hpdict: \n{hp_dict}\n")
#         hp_dict = broadcast_object(hp_dict, master_process)

#         train(cfg, hp_dict, ddp_world_size, master_process)
#     except Exception as e:
#         print(f"Error in ddp_trainer: {e}")
#     finally:
#         cleanup()

# def train(cfg, hp_dict, world_size, master_process):

#     train_loader, val_loader, test_loader = get_dataloaders(cfg["folders"][cfg.task], cfg.preprocessing.image_size, cfg.preprocessing.norm, hp_dict["batch_size"], cfg.testing)
#     model = CustomResNet(num_classes=cfg["classes_per_task"][cfg.task], model_name="resnet50")
#     loss = torch.nn.CrossEntropyLoss(label_smoothing=hp_dict["label_smoothing"])

#     trainer = Trainer(cfg.task, 
#                         model, 
#                         train_loader, 
#                         val_loader, 
#                         loss,
#                         cfg["class_names"][cfg.task],
#                         hp_dict,
#                         world_size=world_size,
#                         master_process=master_process,
#                         testing=cfg.testing,
#                         )
#     trainer.train_model()

# def broadcast_object(obj, master_process):
#     device = f'cuda:{int(os.environ["LOCAL_RANK"])}'
#     if master_process:
#         # Only the source process executes this block
#         obj_bytes = pickle.dumps(obj)
#         obj_size = torch.tensor(len(obj_bytes), dtype=torch.long, device=device)
#     else:
#         obj_size = torch.tensor(0, dtype=torch.long, device=device)

#     # Broadcast the size of the byte stream to all processes
#     print("before broadcast 1")
#     broadcast(obj_size, 0)

#     # Allocate buffer for the object's byte stream
#     obj_bytes = bytearray(obj_size.item())

#     if master_process:
#         # Only the source fills the byte buffer
#         obj_bytes[:] = pickle.dumps(obj)

#     # Create a tensor wrapper for the byte buffer for broadcasting
#     obj_tensor = torch.ByteTensor(obj_bytes).to(device)
#     # Broadcast the byte stream
#     print("before broadcast 2")
#     broadcast(obj_tensor, 0)

#     # Deserialize the byte stream back into the Python object
#     obj = pickle.loads(obj_tensor.cpu().numpy().tobytes())
#     return obj



# def broadcast_config(config_dict, master_process):
    
#     device = f'cuda:{int(os.environ["LOCAL_RANK"])}'
#     local_rank = int(os.environ["LOCAL_RANK"])
#     if master_process:
#         config_json = json.dumps(config_dict)
#         config_bytes = torch.tensor(bytearray(config_json.encode('utf-8')), device=device, dtype=torch.uint8)
#         config_size = torch.tensor([config_bytes.size(0)], device=device)
#     else:
#         print(f"Process {local_rank}: Preparing to receive config...")
#         config_size = torch.tensor([0], device=device)
#         config_bytes = torch.empty(0, device=device, dtype=torch.uint8)

#     # Broadcast the size of the config
#     # print("Before broadcast config_size")
#     # torch.distributed.barrier()
#     broadcast(config_size, src=0)
#     # print("Before second broadcast config_size")
#     # torch.distributed.barrier()
#     # print("After broadcast config_size")

#     if not master_process:
#         config_bytes = torch.empty(config_size.item(), device=device, dtype=torch.uint8)

#     # Synchronize before broadcasting the actual config bytes
#     print("Before broadcast config_bytes")
#     torch.distributed.barrier()
#     broadcast(config_bytes, src=0)
#     print("Before second broadcast config_bytes")
#     torch.distributed.barrier()
#     print("After broadcast config_bytes")

#     if not master_process:
#         # Deserialize the config bytes back into a dictionary
#         config_json = bytes(config_bytes.tolist()).decode('utf-8')
#         print(f"\n5: {device}: {config_json}\n")
#         config_dict = json.loads(config_json)
#         print(f"Process {local_rank}: Received config.")

#     return config_dict

if __name__ == "__main__":
    os.system('clear')
    main()
