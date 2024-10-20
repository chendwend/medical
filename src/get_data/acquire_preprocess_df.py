import sys
from os import system
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))  
from io import BytesIO

import hydra
import pandas as pd
import requests

from utils import verify_dir


def download_df(mass_train_url, mass_test_url, calc_train_url, calc_test_url):
    print("Downloading df...")
    df_mass_train = pd.read_csv(BytesIO(requests.get(mass_train_url).content))
    df_mass_test = pd.read_csv(BytesIO(requests.get(mass_test_url).content))
    df_calc_train = pd.read_csv(BytesIO(requests.get(calc_train_url).content))
    df_calc_test = pd.read_csv(BytesIO(requests.get(calc_test_url).content))
    df_mass, df_calc = pd.concat([df_mass_train, df_mass_test]), pd.concat([df_calc_train, df_calc_test])
    # df_orig = pd.concat([df_mass, df_calc])
    # df = df_orig.copy()
    df = pd.concat([df_mass, df_calc])
    print("Downloading Finished.")
    
    return df

def general_preprocessing(df):
    print("Performing general preprocessing...")
    # fix duplicate column breast density
    df.loc[df['breast_density'].isna(), 'breast_density'] = df.loc[~df['breast density'].isna(), 'breast density']
    df['breast_density'] = df['breast_density'].astype('int32')
    df = df.drop(columns=["breast density"])

    # fix column names
    df.columns = df.columns.str.replace(' ', '_')

    df = df[~df['patient_id'].isin(["P_00016"])] # doesn't have ROI images
    # Benign without callback = benign
    df["pathology"] = df["pathology"].replace(["BENIGN_WITHOUT_CALLBACK"], "BENIGN")

    # unused in training
    df = df.drop(columns=["image_file_path","cropped_image_file_path", "ROI_mask_file_path"])

    # drop calcification cases
    df = df[df["abnormality_type"] == "mass"]

    # convert columns to object type
    df["abnormality_id"] = df["abnormality_id"].astype("object")
    

    return df

def fix_values(df):
  print("Fixing values...")
# Calcification
  df["calc_type"] = df["calc_type"].replace("LUCENT_CENTERED" ,"LUCENT_CENTER")
  df["calc_type"] = df["calc_type"].replace("PLEOMORPHIC-AMORPHOUS" ,"AMORPHOUS-PLEOMORPHIC")
  df["calc_type"] = df["calc_type"].replace("PLEOMORPHIC-PLEOMORPHIC" ,"PLEOMORPHIC")
  df["calc_type"] = df["calc_type"].replace("AMORPHOUS-ROUND_AND_REGULAR" ,"ROUND_AND_REGULAR-AMORPHOUS")
  df["calc_type"] = df["calc_type"].replace("PUNCTATE-LUCENT_CENTER" ,"LUCENT_CENTER-PUNCTATE")
  df["calc_type"] = df["calc_type"].replace("ROUND_AND_REGULAR-LUCENT_CENTERED" ,"ROUND_AND_REGULAR-LUCENT_CENTER")
  df["calc_type"] = df["calc_type"].replace("PUNCTATE-ROUND_AND_REGULAR" ,"ROUND_AND_REGULAR-PUNCTATE")
  df["calc_type"] = df["calc_type"].replace("COARSE-ROUND_AND_REGULAR-LUCENT_CENTERED" ,"COARSE-ROUND_AND_REGULAR-LUCENT_CENTER")

  # Mass
  df["mass_shape"] = df["mass_shape"].replace("LOBULATED-OVAL", "OVAL-LOBULATED")
  df["mass_margins"] = df["mass_margins"].replace("OBSCURED-CIRCUMSCRIBED", "CIRCUMSCRIBED-OBSCURED")

  return df


def remove_low_freq(df):
  print("Removing low frequency values...")
  
  # remove 2 instances of breast_density = 0
  df = df[~df["breast_density"].isin(["0"])]

  return df



def main():
    system("clear")
    
    __dir__ =  Path(__file__).parent.parent.parent
    data_dir =__dir__/"data" 
    verify_dir(data_dir)
    

    with hydra.initialize(version_base="1.3", config_path=""):
        cfg = hydra.compose(config_name="data_conf")

    df = download_df(cfg.mass_train, cfg.mass_test, cfg.calc_train, cfg.calc_test)
    df = general_preprocessing(df)
    df = fix_values(df)
    df = remove_low_freq(df)

    print(f"Saving df to {str(data_dir/'df.csv')}")
    df.to_csv(data_dir/"df.csv")

if __name__ == "__main__":
   main()