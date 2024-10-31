import sys
from os import system
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent)) 
import shutil
from pathlib import Path

import cv2
import hydra
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils import clean_directory, verify_dir


def prep_dirs(__dir__):
    data_dir =__dir__/"data"
    masked_data_dir = data_dir/"masked_cropped"
    dataset_dir = data_dir/"dataset"
    augmented_dir = data_dir/"augmented"
    verify_dir(masked_data_dir, dataset_dir, augmented_dir)

    clean_directory(dataset_dir)

    return masked_data_dir, augmented_dir, dataset_dir

def augment(df, src_dir, augment_dir, rot_angles, clip_limit_values, tileGridSize):
    print("Augmenting training data...")

    for row in tqdm(df.iterrows(), total=len(df), desc="augment progress"):
        src_img = src_dir/ (row[1]["image_filename"] + ".png")
        dest_img = augment_dir / (row[1]["image_filename"] + ".png")
        shutil.copy2(src_img, dest_img)

        with Image.open(dest_img) as img:
            rotation_augment(img, rot_angles, dest_img)
            clip_augment(img, clip_limit_values, tileGridSize, dest_img)
  
def rotation_augment(img, rot_angles, img_path):  
    for rot_angle in rot_angles:
        img_rot = img.rotate(rot_angle, expand=True)
        new_filename = img_path.stem + f"_{rot_angle}" + img_path.suffix 
        dest_path = img_path.with_name(new_filename)
        img_rot.save(dest_path)

def clip_augment(img, clip_limit_values, tileGridSize, img_path):
    for clip_value in clip_limit_values:
        img_np = np.array(img)
        img_np = img_np.astype(np.uint16)
        clahe = cv2.createCLAHE(clipLimit=clip_value, tileGridSize=tileGridSize)
        clahe_image = clahe.apply(img_np)
        clahe_pil_image = Image.fromarray(clahe_image)

        new_filename = img_path.stem + f"_clahe{clip_value}" + img_path.suffix 
        dest_path = img_path.with_name(new_filename)
        clahe_pil_image.save(dest_path)

def split_df(task, df, test_size, val_size , seed):

    print(f"splitting training data for {task} task...")

    patient_groups = df.groupby('patient_id')['pathology'].agg(lambda x: x.mode()[0]).reset_index()
    patient_groups = patient_groups.rename(columns={'pathology': 'stratify_label'})

    train_val_ids, test_ids = train_test_split(
        patient_groups['patient_id'],
        test_size=test_size,
        stratify=patient_groups['stratify_label'],
        random_state=seed
    )
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=val_size,
        stratify=patient_groups[patient_groups['patient_id'].isin(train_val_ids)]['stratify_label'],
        random_state=seed
    )
    
    train_df = df[df["patient_id"].isin(train_ids)]
    val_df = df[df["patient_id"].isin(val_ids)]
    test_df = df[df["patient_id"].isin(test_ids)]

    return train_df, val_df, test_df

def generate_dataset_according_to_task(task, train_df, val_df, test_df, src_dir, dataset_dir, augment_dir, rot_angles, clip_limit_values):

    for df_type, split in zip([val_df, test_df], ['val', 'test']):
    
        for row in tqdm(df_type.iterrows(), total=len(df_type), desc=f"{split} processing..."):
            filename = row[1]["image_filename"] + ".png"
            src_file = src_dir/filename

            dest_dir = dataset_dir/task/split/str(row[1][task])
            verify_dir(dest_dir, notify=False)
            dest_file = dest_dir/ src_file.name
            shutil.copy2(src_file, dest_file)
    
    for row in tqdm(train_df.iterrows(), total=len(train_df), desc=f"train processing..."):
        filename = row[1]["image_filename"] + ".png"
        src_file = augment_dir/filename
        dest_dir = dataset_dir/task/"train"/str(row[1][task])
        verify_dir(dest_dir, notify=False)
        dest_file = dest_dir/ src_file.name
        shutil.copy2(src_file, dest_file)
        for rot_angle in rot_angles:
            filename = row[1]["image_filename"] + f"_{rot_angle}" + ".png"
            src_file = augment_dir/filename
            dest_file = dest_dir/ src_file.name
            shutil.copy2(src_file, dest_file)
        for clip_val in clip_limit_values:
            filename = row[1]["image_filename"] + f"_clahe{clip_val}" + ".png"
            src_file = augment_dir/filename
            dest_file = dest_dir/ src_file.name
            shutil.copy2(src_file, dest_file)

def main():
    with hydra.initialize(version_base="1.3", config_path=""):
        cfg = hydra.compose(config_name="data_conf")

    __dir__ = Path(__file__).parent.parent.parent
    src_dir, augment_dir, dataset_dir = prep_dirs(__dir__)
    df_orig = pd.read_csv(__dir__/"data"/"df.csv", index_col=0)
    df_orig["image_filename"] = df_orig[[
        "patient_id", 
        "abnormality_type", 
        "left_or_right_breast", 
        "image_view", 
        "abnormality_id"
        ]].astype(str).apply("_".join, axis=1)

    for task in ["assessment", "pathology", "mass_shape", "breast_density"]:
        print(f"Generating training data for {task} task...")
        df = df_orig.query(f"{task} in @cfg.{task}")
        train_df, val_df, test_df = split_df(task, df, cfg.test_size, cfg.val_size, cfg.seed)
        clean_directory(augment_dir, notify=False)
        augment(train_df, src_dir, augment_dir, cfg.rot_angles, cfg.clip_limit_values, cfg.tileGridSize)
        generate_dataset_according_to_task(task, train_df, val_df, test_df, src_dir, dataset_dir, augment_dir, 
                                           cfg.rot_angles, cfg.clip_limit_values)

    print("Finished generating training data.")

    

if __name__ == "__main__":
    main()



    