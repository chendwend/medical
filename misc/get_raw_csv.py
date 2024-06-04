from io import BytesIO
from pathlib import Path

import hydra
import pandas as pd
import requests
from utils import create_folder


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg):

    create_folder(cfg.folders.csv_files.raw)

    df_mass_train = pd.read_csv(BytesIO(requests.get(cfg.urls.mass_train).content))
    df_mass_test = pd.read_csv(BytesIO(requests.get(cfg.urls.mass_test).content))
    df_calc_train = pd.read_csv(BytesIO(requests.get(cfg.urls.calc_train).content))
    df_calc_test = pd.read_csv(BytesIO(requests.get(cfg.urls.calc_test).content))
    df_mass, df_calc = pd.concat([df_mass_train, df_mass_test]), pd.concat([df_calc_train, df_calc_test])
    df = pd.concat([df_mass, df_calc])

    df.to_csv(Path(cfg.folders.csv_files.raw)/"df.csv")
    df_mass.to_csv(Path(cfg.folders.csv_files.raw)/"df_mass.csv")
    df_calc.to_csv(Path(cfg.folders.csv_files.raw)/"df_calc.csv")

if __name__ == '__main__':
    main()