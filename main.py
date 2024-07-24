import os

import hydra
import torch
import wandb
from dotenv import load_dotenv
from torch.distributed import destroy_process_group, init_process_group

from src.dataloaders import get_dataloaders
from src.model import CustomResNet
from src.Trainer import Trainer


def setup(seed:int, cfg:dict) -> bool:
    """ 
    Performs all neccesary setup

    Args:
        seed (int): seed for setting randomization

    Returns:
        bool: Indication whether ddp training.
    """
    def setup_wandb(master_process, hp_dict):
        if master_process:
            api_key = os.getenv("WANDB_API_KEY")
            if not api_key:
                raise ValueError("WANDB_API_KEY is not set in the environment variables.")
            
            run = wandb.init(project="medical", config=hp_dict)
        else:
            run = None

        return run


    def ddp_setup() -> None:

        init_process_group(backend="nccl")
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        


    load_dotenv()
    

    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        ddp_setup()
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        master_process = ddp_local_rank == 0
        print("Initiating distributed training...")
    else:
        master_process = True
        ddp_world_size = 1

    run = setup_wandb(master_process, hp_dict)



    torch.manual_seed(seed+ddp)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Computation device: {DEVICE}\n")
    
    return ddp, master_process, ddp_world_size, run


def cleanup(ddp):
    if ddp:
        destroy_process_group()
    torch.cuda.empty_cache()


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg):
    ddp, master_process, world_size, wandb_run = setup(cfg.seed)

    train_loader, val_loader, test_loader = get_dataloaders(cfg, cfg.task, ddp, cfg.testing)

    model = CustomResNet(num_classes=cfg["classes_per_task"][cfg.task], model_name="resnet50")

    loss = torch.nn.CrossEntropyLoss(label_smoothing=cfg.hp.label_smoothing)

    trainer = Trainer(cfg.task, 
                    model, 
                    train_loader, 
                    val_loader, 
                    loss,
                    cfg["class_names"][cfg.task],
                    cfg.hp,
                    world_size=world_size,
                    master_process=master_process,
                    wandb_run = wandb_run,
                    test=cfg.testing,
                    )
    trainer.train_model()
    cleanup(ddp)

       

if __name__ == "__main__":
    os.system('clear')
    main()
