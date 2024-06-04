from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from utils import create_folder


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg):

    create_folder(cfg.folders.csv_files.preprocessed)

    df = pd.read_csv(Path(cfg.folders.csv_files.raw)/"df.csv", index_col=0)
    df_mass = pd.read_csv(Path(cfg.folders.csv_files.raw)/"df_mass.csv", index_col=0)
    df_calc = pd.read_csv(Path(cfg.folders.csv_files.raw)/"df_calc.csv", index_col=0)

    # fix duplicate column breast density
    df.loc[df['breast_density'].isna(), 'breast_density'] = df.loc[~df['breast density'].isna(), 'breast density']
    df['breast_density'] = df['breast_density'].astype('int32')
    df = df.drop(columns=['breast density'])
    df_mass.columns, df_calc.columns  = df_mass.columns.str.replace(' ', '_'), df_calc.columns.str.replace(' ', '_')
    # all columns with space  = > underscore
    df.columns = df.columns.str.replace(' ', '_')
    # Fix image paths
    df['image_file_name'] = df['image_file_path'].str.split('/').str[-1]
    df['PatientID'] = df['image_file_path'].str.split('/').str[0]
    df['image_file_path'] = df['image_file_path'].str.split('/').str[-2]
    df['cropped_image_file_name'] = df['cropped_image_file_path'].str.split('/').str[-1]
    df['cropped_image_file_path'] = df['cropped_image_file_path'].str.split('/').str[-2]
    df['ROI_mask_file_name'] = df['ROI_mask_file_path'].str.split('/').str[-1]
    df['ROI_mask_file_path'] = df['ROI_mask_file_path'].str.split('/').str[-2]

    for folder in ["full_folder", "roi_folder", "mask_folder"]:
        df[folder] = df["patient_id"] + "_" +folder.split("_")[0] + "_" + df["left_or_right_breast"] + "_" + df["image_view"] \
        + "_" + df["abnormality_type"] + "_" + df["abnormality_id"].astype("str")

    df['cropped_image_file_name'] = df['cropped_image_file_name'].replace({'000000.dcm\n': '000000.dcm', '000001.dcm\n': '000001.dcm'})
    df['ROI_mask_file_name'] = df['ROI_mask_file_name'].replace({'000000.dcm\n':'000000.dcm', '000001.dcm\n':'000001.dcm'})

    condlist = [
    (df["ROI_mask_file_name"] == "000000.dcm") &  (df["cropped_image_file_name"] == "000000.dcm"),
        (df["ROI_mask_file_name"] == "000000.dcm") &  (df["cropped_image_file_name"] == "000001.dcm"),
        (df["ROI_mask_file_name"] == "000001.dcm") &  (df["cropped_image_file_name"] == "000000.dcm")
    ]
    choicelist_roi_mask = [
        "1-1.dcm",
        "1-2.dcm",
        "1-1.dcm"
    ]

    choicelist_cropped = [
        "1-1.dcm",
        "1-1.dcm",
        "1-2.dcm"
    ]


    df['ROI_mask_file_name_true'] = np.select(condlist, choicelist_roi_mask)
    df['cropped_image_file_name_true'] = np.select(condlist, choicelist_cropped)

    # save preprocessed csv
    df.to_csv(Path(cfg.folders.csv_files.preprocessed)/"df.csv")
    df_mass.to_csv(Path(cfg.folders.csv_files.preprocessed)/"df_mass.csv")
    df_calc.to_csv(Path(cfg.folders.csv_files.preprocessed)/"df_calc.csv")

if __name__ == '__main__':
    main()
