import os
import torch
import torch.distributed as dist
import wandb
import pickle
from dataloaders import get_dataloaders
from model import CustomResNet
from Trainer import Trainer
import hydra


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

def setup(rank, seed):
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    torch.manual_seed(seed) 

def cleanup():
    dist.destroy_process_group()
    torch.cuda.empty_cache()

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def train(cfg):

    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    master_process = rank == 0
    try:
        setup(rank, cfg["seed"])

        if rank == 0:
            wandb.init()
            config = wandb.config
            hp_dict = {k: v for k, v in dict(config).items()}
            param_string = "_".join([f"{k}={v}" for k, v in hp_dict.items()])
            wandb.run.name = param_string
            # print(f"{master_process}: {hp_dict}")
        else:
            hp_dict = None

        hp_dict = broadcast_object(hp_dict, rank==0)
        torch.cuda.set_device(rank)

        train_loader, val_loader, test_loader = get_dataloaders(cfg["folders"][cfg.task], cfg.preprocessing.image_size, cfg.preprocessing.norm, hp_dict["batch_size"], cfg.testing)
        model = CustomResNet(num_classes=cfg["classes_per_task"][cfg.task], model_name="resnet50", fc_layer=hp_dict["fc_layer"])
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
    finally:
        cleanup()

if __name__ == "__main__":
    train()