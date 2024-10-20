import argparse
import os
import subprocess

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

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


def run_training(gpu_devices:str, gpu_device_count:int, task:str, model:str, workers:bool=False, nnodes:int=1):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ", ".join([str(a) for a in gpu_devices])
    command = [
        "torchrun",
        f"--nproc_per_node={str(gpu_device_count)}",
        f"--nnodes={str(nnodes)}",
        "src/train_ddp.py",
        f"--model={model}",
        f"--task={task}"
    ]

    if workers:
        command.append("-w")

    result = subprocess.run(command, env=env, check=True)
    if result.returncode == 0:
        print("Sweep finished Successfully")
    else:
        print("------Sweep FAILED------")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--model" , help="model type",  choices=["resnet50", "resnet101", "resnet152"])
    parser.add_argument("-t", "--task", help="type of task", choices=["pathology", "birads", "mass_shape"])
    parser.add_argument("-d", "--devices", nargs="+", type=str, help="list of selected devices", default="all")
    parser.add_argument("-w", "--workers", help="test num_workes", action="store_true")

    args = parser.parse_args()

    if args.devices == "all":
        from torch.cuda import device_count
        args.devices = [x for x in range(device_count())]

    args.num_devices = len(args.devices)
    
    return args


def main(cfg:DictConfig):
    if not cfg.workers:
        sweep_id, sweep_count = setup_wandb(cfg.sweep_counts)
        print(f"Initiating with sweep counts = {sweep_count}.")
        wandb.agent(sweep_id, 
                    function=lambda: run_training(cfg.gpu_devices, 
                                                  cfg.gpu_device_count, 
                                                  cfg.task, 
                                                  cfg.model), 
                    count=sweep_count, 
                    project="medical")
    else:
        run_training(cfg.gpu_devices, 
                     cfg.gpu_device_count, 
                     cfg.task, 
                     cfg.model, 
                     cfg.workers,
                     1)


if __name__ == "__main__":
    os.system('clear')
    
    args = parse_args()

    overrides = [
        f"model={args.model}",
        f"gpu_devices={args.devices}",
        f"gpu_device_count={args.num_devices}",
        f"task={args.task}",
        f"workers={args.workers}"
    ]

    with hydra.initialize(version_base="1.3", config_path="conf"):
        cfg = hydra.compose(config_name="config", overrides=overrides)
        main(cfg)
