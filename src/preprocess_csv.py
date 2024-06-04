from os import chdir
from pathlib import Path

import hydra
import pandas as pd

from utils import chngdir


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg):
    chngdir()

    df = pd.read_csv(Path(cfg.folders.data)/cfg.csv.raw)
    df_orig = df.copy()

    # fix image column name
    df = df.rename(columns={'Image':'image_url'})


    # drop duplicate image urls
    df = df.drop_duplicates(subset="image_url")
    duplicate_image_urls = len(df_orig) - len(df)
    cur_len = len(df)

    # Drop irrelevant columns
    drop_columns = ["Acq name (previous)","Acq name (acq)", "Acq date", "Acq notes (exc)", "BM/Big number",
                  "Add ids", "Location","Dept", "Museum number", "Culture", "Reg number", "Bib references",
                  "Exhibition history", "Condition"]
    df = df.drop(columns=drop_columns)

    # remove columns with #NANs above threshold
    df = df.loc[:, df.isnull().mean() < cfg.nan_threshold]
    # remove NaNs in Authority
    df = df.dropna(subset="Authority")
    df = df.reset_index().drop(columns='index')
    nan_rows_dropped = cur_len - len(df)
    cur_len = len(df)

    # Remove ambivalent authority
    df = df.query("Authority!='Sennacherib; Ashurbanipal' and Authority !='Sennacherib; Ashurbanipal; Sin-shar-ishkun'")
    ambivalent_authority_dropped = cur_len - len(df)
    cur_len = len(df)


    # Save processed df to csv
    df.to_csv(Path(cfg.folders.data)/cfg.csv.processed, index=False)

    total_rows_dropped = nan_rows_dropped+duplicate_image_urls + ambivalent_authority_dropped
    print(f"Original dataset {len(df_orig)} rows.")
    print(f"Dropped a total of {total_rows_dropped} images, {total_rows_dropped/len(df_orig)*100:.2f}%")
    print(f"Of which {nan_rows_dropped} NaNs, {duplicate_image_urls} duplicate image urls and " 
          f"{ambivalent_authority_dropped} ambivalent authority")
    print(f"Total images for model: {len(df)}")
    
if __name__ == '__main__':
    main()

 





