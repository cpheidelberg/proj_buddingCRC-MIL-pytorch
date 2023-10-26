from tqdm import tqdm
import os
import numpy as np

def adapt_file_names(df, sds_path, keys, wanna_check):


    if isinstance(keys, str):
        keys = [keys]
    def replace_string(nested_list, substring_to_replace, replacement_string):

      nested_list = nested_list.to_list()
      for i in tqdm(range(0, len(nested_list)), desc="1st round file name adapation"):
        sub_list = nested_list[i]
        for j in range(0, len(sub_list)):
          sub_list[j] = sub_list[j].replace(substring_to_replace, replacement_string)
        nested_list[i] = sub_list

      return nested_list

    substring_to_replace = "/home/usr/root/"
    replacement_string = sds_path
    for i_key in keys:
        df[i_key] = replace_string(df[i_key], substring_to_replace, replacement_string)

    if wanna_check:

        def check_file_existence(file_names):

            indices = []
            for idx in range(0, len(file_names)):
                try:
                    img = Image.open(file_names[idx])
                    img.close()
                    indices.append(idx)
                except (IOError, OSError):
                    print(f"Unable to load image: {file_names[idx]}")
            indices = np.array(indices)
            existing_files = [file_names[i] for i in indices]
            return existing_files

        def replace_string(nested_list, substring_to_replace, replacement_string):

          nested_list = nested_list.to_list()
          nested_list_cleaned = []
          for i in tqdm(range(0, len(nested_list)), desc="2nd round file name adaption"):
            sub_list = nested_list[i]
            for j in range(0, len(sub_list)):
              sub_list[j] = sub_list[j].replace(substring_to_replace, replacement_string)
            nested_list_cleaned.append(check_file_existence(sub_list))

          return nested_list_cleaned

    substring_to_replace = "GrazKollektiv"
    replacement_string = "GrazKollektiv/Original-Kollektiv"

    for i_key in keys:
        df[i_key] = replace_string(df[i_key], substring_to_replace, replacement_string)

    substring_to_replace = "Original-Kollektiv/Original-Kollektiv"
    replacement_string = "Original-Kollektiv"

    for i_key in keys:
        df[i_key] = replace_string(df[i_key], substring_to_replace, replacement_string)

    return df


#%% clean up
from PIL import Image

def check_image_file(image_path):

    try:
        img = Image.open(image_path)
        img.close()
    except (IOError, OSError):
        print(f"Unable to load image: {image_path}")
        return False
    return True

def check_image_files(image_paths):

    good_files = []
    for idx, i_file in enumerate(image_paths):
        if check_image_file(i_file):
            good_files.append(idx)

    image_paths = [image_paths[i] for i in good_files]

    return image_paths

import pandas as pd
def remove_rows_by_indices(df, indices):
    df = df.drop(indices)
    df.reset_index(drop=True, inplace=True)
    return df

def check_colum(colum_name, df):

    indices = []
    for i in range(0, len(df)):
        if check_image_files(df[colum_name][i]):
            indices.append(i)

    df = remove_rows_by_indices(df, indices)

    return df