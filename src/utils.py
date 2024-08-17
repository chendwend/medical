import shutil
from os import chdir
from pathlib import Path

import requests


def chngdir():
    print(f"cwd:{Path().absolute()}")
    # script_dir = Path(__file__).parent
    # print(f"script parent directory: {script_dir}")
    # chdir(script_dir)
    # print(f"cwd after changing to script directory:{Path().absolute()}")


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


def clean_directory(path):
    """
    Recursively delete all files and subdirectories in a given directory.
    
    :param path: Pathlib Path object or string of the directory to clean.
    """
    dir_path = Path(path)
    if dir_path.exists() and dir_path.is_dir():
        for item in dir_path.iterdir():
            if item.is_dir():
                shutil.rmtree(item)  # Recursively delete directory
            else:
                item.unlink()  # Delete file
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