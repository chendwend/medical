import shutil
from pathlib import Path

import requests


def download_image(url, directory, label, index):
    image_name = f"{label}_{index}.jpg"
    image_path = Path(directory, label, image_name)

    if not image_path.exists():
        response = requests.get(url, verify=False)
        if response.status_code == 200:
            with open(image_path, "wb") as f:
                f.write(response.content)

    return image_path


def remove_dir(dir_path) -> None:
    dirpath = Path(dir_path)

    # Check if the directory exists
    if dirpath.exists():
        # Delete the directory, even if it is not empty
        shutil.rmtree(dirpath)

def clean_directory(*dirs, notify=True):
    """
    Recursively delete all files and subdirectories in directory or directories.
    
    :param path: Pathlib Path object or string of the directory to clean.
    """
    for directory in dirs:
        dir_path = Path(directory)
        if dir_path.exists() and dir_path.is_dir():
            for item in dir_path.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)  # Recursively delete directory
                else:
                    item.unlink()  # Delete file
            if notify:
                print(f"All contents removed from {dir_path}")
        else:
            print(f"The directory {dir_path} does not exist or is not a directory.")

def time_it(func):
    from time import perf_counter
    def wrapper(*args, **kwargs):
        start_time = perf_counter()
        result = func(*args, **kwargs)
        end_time = perf_counter()
        run_time = end_time - start_time
        return result, run_time
    return wrapper

def verify_dir(*dirs, notify=True):
    for directory in dirs:
        directory = Path(directory)
        if not directory.exists():
            directory.mkdir(exist_ok=True, parents=True)
            if notify:
                print(f"{str(directory)} was created.")

def seed_everything(base_seed:int, rank:int):
    import random

    import numpy as np
    import torch


    seed = base_seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
