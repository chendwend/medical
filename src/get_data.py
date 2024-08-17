import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils import clean_directory


def create_dataset(task, clip_limit_values=[2,10], train_size=0.8, val_size=0.5):
    rot_angles = [90, 180, 270]
    tileGridSize=(8, 8)
    birads_val = [0, 3, 4, 5]
    seed = 42

    __dir__ = Path().absolute()
    data_dir =__dir__/"data" 
    if not(data_dir.is_dir()):
        raise(f"{data_dir} directory doesn't exist!")
    masked_data_dir = data_dir/"masked_cropped"
    dataset_dir = data_dir/"dataset"
    augmented_dir = data_dir/"augmented"

    masked_data_dir.mkdir(exist_ok=True)
    augmented_dir.mkdir(exist_ok=True)
    dataset_dir.mkdir(exist_ok=True)


    df = pd.read_csv(data_dir/"df.csv", index_col=0)
    # df_meta = pd.read_csv(data_dir/"df_meta.csv",index_col=0)

    clean_directory(augmented_dir)
    src_paths = list(masked_data_dir.glob("*.png"))
    for img_path in tqdm(src_paths, total=len(src_paths), desc="Augmenting x6..."):
        with Image.open(img_path) as img:
            for rot_angle in rot_angles:
                img_rot = img.rotate(rot_angle, expand=True)
                dest_path = augmented_dir/(img_path.stem + f"_{str(rot_angle)}.png")
                img_rot.save(dest_path)

            for clip_value in clip_limit_values:
                img_np = np.array(img)
                img_np = img_np.astype(np.uint16)
                clahe = cv2.createCLAHE(clipLimit=clip_value, tileGridSize=tileGridSize)
                clahe_image = clahe.apply(img_np)
                clahe_pil_image = Image.fromarray(clahe_image)

                dest_path = augmented_dir/(img_path.stem + f"_clahe{clip_value}.png")
                clahe_pil_image.save(dest_path)
            dest_file = augmented_dir/img_path.name
            shutil.copy2(img_path, dest_file)

    # Create augmented df
    df_augmented = df.copy()
    for rot_angle in rot_angles:
        df_rot = df.copy()
        df_rot['masked_cropped'] = df_rot['masked_cropped'] + f'_{str(rot_angle)}'
        df_augmented = pd.concat([df_augmented, df_rot], ignore_index=True)

    for clip_value in clip_limit_values:
        df_clahe = df.copy()
        df_clahe['masked_cropped'] = df_clahe['masked_cropped'] + f'_clahe{clip_value}'
        df_augmented = pd.concat([df_augmented, df_clahe], ignore_index=True)


    # Split train/val/test
    clean_directory(dataset_dir)
    df_augmented = df_augmented.sample(frac=1, random_state=seed).reset_index(drop=True)
    for task in ["assessment", "pathology", "mass_shape"]:
        train_df, temp_df = train_test_split(df_augmented, train_size=train_size, random_state=seed, stratify=df_augmented[task])
        val_df, test_df = train_test_split(temp_df, train_size=val_size, random_state=seed, stratify=temp_df[task])
        print(f"working on {task}...")
        for df_type, split in zip([train_df, val_df, test_df], ['train', 'val', 'test']):
            
            for row in tqdm(df_type.iterrows(), total=len(df_type), desc=f"{split} processing..."):
                filename = row[1]["masked_cropped"] + ".png"
                src_file = augmented_dir/filename

                dest_dir = dataset_dir/task/split/str(row[1][task])
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_file = dest_dir/ src_file.name
                shutil.copy2(src_file, dest_file)



