import os
import hydra
from dotenv import load_dotenv
import wandb
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
                # 'values': [2]
                'values': [30]
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
                'values': [32, 64, 128]
                # 'values': [32]
            },
            'label_smoothing':{
                'values':  [0.25, 0.35]
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project='medical')
    return sweep_id

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

if __name__ == "__main__":
    os.system('clear')
    main()
