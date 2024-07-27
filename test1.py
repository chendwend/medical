import wandb
import subprocess
from dotenv import load_dotenv
from os import system

def run_training():
    
    wandb.init()
    subprocess.run(["torchrun", "--nproc_per_node=2", "--nnodes=1", "src/train_ddp.py"])

if __name__ == "__main__":
    system("clear")
    sweep_config = {
        'method': 'random',
        'metric': {'name': 'loss', 'goal': 'minimize'},
        'parameters': {
            'lr': {'min': 0.001, 'max': 0.1},
            'epochs': {'values': [5, 10, 15]}
        }
    }
    load_dotenv()
    sweep_id = wandb.sweep(sweep_config, project="ddp_wandb_example")
    wandb.agent(sweep_id, function=run_training, count=5)