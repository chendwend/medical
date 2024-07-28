import wandb
import subprocess
from dotenv import load_dotenv
from os import system

def run_training():
    
    wandb.init()
    subprocess.run(["torchrun", "--nproc_per_node=2", "--nnodes=1", "src/train_ddp_test.py"])

if __name__ == "__main__":
    system("clear")
    # sweep_config = {
    #     'method': 'random',
    #     'metric': {'name': 'loss', 'goal': 'minimize'},
    #     'parameters': {
    #         'lr': {'min': 0.001, 'max': 0.1},
    #         'epochs': {'values': [5, 10, 15]}
    #     }
    # }
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
    load_dotenv()
    sweep_id = wandb.sweep(sweep_config, project="ddp_wandb_example")
    wandb.agent(sweep_id, function=run_training, count=5)