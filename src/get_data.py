from pathlib import Path
import pandas as pd, numpy as np
from tqdm import tqdm
import pydicom as dicom
from io import BytesIO
import requests
from tcia_utils import nbia
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import math
from sklearn.model_selection import train_test_split
import shutil
import torch

__dir__ = Path().absolute()
data_dir =__dir__/"data" 
# dcm_dir = data_dir/"dcm"
masked_data_dir = data_dir/"masked_cropped"
dataset_dir = data_dir/"dataset"
augmented_dir = data_dir/"augmented"

# dcm_dir.mkdir(exist_ok=True)
masked_data_dir.mkdir(exist_ok=True)
augmented_dir.mkdir(exist_ok=True)
dataset_dir.mkdir(exist_ok=True)

rot_angles = [90, 180, 270]
clip_limit_values = [2, 10]
tileGridSize=(8, 8)


dcm_extension = ".dcm"

df = pd.read_csv(data_dir/"df.csv", index_col=0)
df_meta = pd.read_csv(data_dir/"df_meta.csv",index_col=0)

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


# preprocessing
# Remove unique value columns
# df_meta = df_meta[df_meta.columns[df_meta.nunique() > 1]]

# mutual preprocessing
# df_meta_orig = df_meta_orig[df_meta_orig.columns[df_meta_orig.nunique() > 1]]
# df_meta_orig['PatientID_num'] = df_meta_orig['PatientID'].str.split('_').str[1:3].str.join('_')
# df_meta_orig["abnormality_type"] = df_meta_orig['PatientID'].str.split('-').str[0]


# df_meta['PatientID_num'] = df_meta['PatientID'].str.split('_').str[1:3].str.join('_')
# # df_meta_orig['PatientID_num'] = df_meta_orig['PatientID'].str.split('_').str[1:3].str.join('_')
# df_meta["abnormality_type"] = df_meta['PatientID'].str.split('-').str[0]
# # df_meta_orig["abnormality_type"] = df_meta_orig['PatientID'].str.split('-').str[0]
# df_meta = df_meta[df_meta['abnormality_type'] == "Mass"]
# df_meta['PatientID'] = df_meta['PatientID'].str.split('_').str[3:].str.join("_") 
# df_meta["abnorm_num"] = df_meta['PatientID'].str.split('_').str[2]
# df_meta['abnorm_num'] = df_meta['abnorm_num'].fillna(1) # all NaN are only 1 abnormality
# df_meta['PatientID'] = df_meta['PatientID'].str.split('_').str[0:2].str.join("_")

# fix duplicate column breast density
# df.loc[df['breast_density'].isna(), 'breast_density'] = df.loc[~df['breast density'].isna(), 'breast density']
# df['breast_density'] = df['breast_density'].astype('int32')
# df = df.drop(columns=["breast density"])

# # fix column names
# df_mass.columns, df_calc.columns  = df_mass.columns.str.replace(' ', '_'), df_calc.columns.str.replace(' ', '_')
# df.columns = df.columns.str.replace(' ', '_')

# # mass shape limited to most common
# mass_shape_val = ['OVAL', 'IRREGULAR', 'LOBULATED', 'ROUND']
# df = df.query("mass_shape in @mass_shape_val")

# assessment limited to 0,3,4,5
birads_val = [0, 3, 4, 5]
# df = df.query("assessment in @birads_val")

# # create masked_cropped column for aiding in dataset split
# df["masked_cropped"] = df["patient_id"] + "_" + "Mass" + "_" + df["left_or_right_breast"] + "_" + df["image_view"] + "_" +df["abnormality_id"].astype('str')

# # Benign without callback = benign
# df["pathology"] = df["pathology"].replace(["BENIGN_WITHOUT_CALLBACK"], "BENIGN")

# # drop unused columns
# df_meta = df_meta.drop(columns=["StudyInstanceUID", "TimeStamp", "FileSize", "abnormality_type"])
# df = df.drop(columns=["image_file_path","cropped_image_file_path", "ROI_mask_file_path"])

# df = df[~df['patient_id'].isin(["P_00016"])] # doesn't have ROI images

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
train_size = 0.8
val_size = 0.5
seed = 42

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



