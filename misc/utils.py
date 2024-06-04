from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pydicom as dicom
from tcia_utils import nbia

# Configs
dcm_extension = ".dcm"

def create_folder(path):
  __dir__ = Path(path)
  __dir__.mkdir(parents=True, exist_ok=True)



def showNanSummary(df, precision=2):
    nan_count = df.isna().sum()
    nan_percentage = round((nan_count / df.shape[0]) * 100, precision)
    nan_summary = pd.DataFrame({'NaN Count': nan_count, 'NaN Percentage': nan_percentage})
    nan_summary = nan_summary[nan_summary['NaN Count'] > 0]  # Exclude columns with no NaNs
    print(nan_summary)

def get_patientID_info(df, id):
  """get all patient records from df"""
  print(df.query("patient_id == @id"))

def download_patient(data_dir, df, id):
  """download all data given patient id"""
  patient_dir = data_dir/id
  patient_dir.mkdir(exist_ok=True)
  series_uids = df.query("PatientID_num == @id")['SeriesInstanceUID'].to_list()
  nbia.downloadSeries(series_uids, input_type = "list", path=str(patient_dir))
  return patient_dir

def map_patient_subfolders(df, id_folder, id):
  """rename all subfolders of given patient id folder to english"""
  df_patient = df.query("patient_id==@id")

  for full_orig, cropped_orig,                                                   \
      mask_orig, full_new,                                                   \
      cropped_new, mask_new,                                                    \
      ROI_mask_file_name_true, cropped_image_file_name_true in              \
   zip(df_patient["image_file_path"], df_patient["cropped_image_file_path"], \
       df_patient["ROI_mask_file_path"], df_patient["full_folder"],          \
       df_patient["roi_folder"], df_patient["mask_folder"],
       df_patient["ROI_mask_file_name_true"], df_patient["cropped_image_file_name_true"]):

    orig_full_folder_path = id_folder/full_orig
    orig_cropped_folder_path = id_folder/cropped_orig
    orig_mask_folder_path = id_folder/mask_orig

    new_full_folder_path = id_folder/full_new
    new_cropped_folder_path = id_folder/cropped_new
    new_mask_folder_path = id_folder/mask_new

    orig_full_image_path = new_full_folder_path/"1-1.dcm"
    orig_cropped_image_path  = new_cropped_folder_path/ cropped_image_file_name_true
    orig_mask_image_path = new_mask_folder_path / ROI_mask_file_name_true

    try:
      orig_full_folder_path.rename(orig_full_folder_path.parent/full_new)
      orig_full_image_path.rename(orig_full_image_path.parent/(full_new+dcm_extension))
    except FileNotFoundError:
      # full mm image already renamed
      pass

    orig_cropped_folder_path.rename(orig_cropped_folder_path.parent/cropped_new)
    orig_cropped_image_path.rename(orig_cropped_image_path.parent/(cropped_new+dcm_extension))
    try:
      orig_mask_folder_path.rename(orig_mask_folder_path.parent/mask_new)
      orig_mask_image_path.rename(orig_mask_image_path.parent / (mask_new + dcm_extension))
    except FileNotFoundError:
      orig_mask_folder_path = new_cropped_folder_path
      orig_mask_image_path = orig_mask_folder_path/ ROI_mask_file_name_true
      orig_mask_image_path.rename(orig_mask_image_path.parent / (mask_new + dcm_extension))

def download_arange_patient(id):
  """download and arange patient folder"""
  patient_dir = download_patient(id)
  map_patient_subfolders(patient_dir, id)

# def read_dcm():
def display_dcm(data_dir, id):
  folder_path = data_dir/id
  dcm_files = list(folder_path.rglob("*.dcm"))
  for dcm in dcm_files:
    ds1 = dicom.dcmread(str(folder_path/dcm))
    a = ds1.pixel_array
    plt.figure()
    plt.imshow(a, cmap='gray')
    plt.xticks([]), plt.yticks([])
    title = dcm.name.split(".")[0]
    plt.title(title)