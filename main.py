import os
import subprocess

import hydra
from dotenv import load_dotenv

import wandb


def setup_wandb(random_sweep_count):
    import yaml
    from torch.cuda import device_count

    load_dotenv()
    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        raise ValueError("WANDB_API_KEY is not set in the environment variables.")
    
    num_devices = device_count()
    with open("conf/sweep_config.yaml") as stream:
        try:
            sweep_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise("conf/sweep_config.yaml not found!")
    sweep_config['parameters']['batch_size']['values'][0] = num_devices * sweep_config['parameters']['batch_size']['values'][0]
   
    if sweep_config["method"] == 'grid':
        sweep_count = calculate_combinations(sweep_config['parameters'])
    elif sweep_config["method"] == 'random':
        sweep_count = random_sweep_count

    sweep_id = wandb.sweep(sweep_config, project='medical')

    return sweep_id, sweep_count

def calculate_combinations(parameters:dict) -> int:
    import itertools

    values = [v['values'] for v in parameters.values()]
    return len(list(itertools.product(*values)))


def run_training():
    from torch.cuda import device_count

    num_gpus = device_count()
    
    wandb.init()
    subprocess.run(["torchrun", f"--nproc_per_node={num_gpus}", "--nnodes=1", "src/train_ddp.py"])

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg):

    sweep_id, sweep_count = setup_wandb(cfg.sweep_counts)
    print(f"Initiating with sweep counts = {sweep_count}.")
    wandb.agent(sweep_id, function=run_training, count=sweep_count, project="medical")

if __name__ == "__main__":
    os.system('clear')
    main()
