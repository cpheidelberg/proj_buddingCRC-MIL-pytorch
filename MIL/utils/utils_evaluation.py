import os
import shutil

import torch


#%% define folder preparation
def create_clean_folder(folder_path):
    # Check if the folder already exists
    if os.path.exists(folder_path):
        # If it exists, remove the folder and its contents
        shutil.rmtree(folder_path)

    # Create the new folder
    os.makedirs(folder_path)

#%% define file name preparation
class FileList():
    def __init__(self, file_list, sds_path):
        self.file_list = file_list
        self.sds_path = sds_path

    def __getitem__(self, index):
        file_name_original = self.file_list[index]
        file_name = file_name_original.replace("/home/usr/root/dir/", self.sds_path + "dirx/")
        return file_name

#%% reset the labels
import numpy as np
import torch
def reset_label(ids):

    unique_label = np.unique(ids)
    ids_new = np.zeros(ids.shape)
    for i, i_label in enumerate(unique_label):
        idx = ids == i_label
        ids_new[idx] = i

    return torch.from_numpy(ids_new)

#%% get the statistics
def get_boxplot_statistics(dataframe, column_name = "prob"):
    """
    Calculate and return boxplot statistics for a specified column in a Pandas DataFrame.

    Parameters:
    - dataframe: The Pandas DataFrame containing the data.
    - column_name: The name of the column for which to calculate statistics.

    Returns:
    A dictionary containing the following boxplot statistics:
    {
        'mean': Mean value,
        'median': Median value,
        '25th_percentile': 25th percentile value,
        '75th_percentile': 75th percentile value,
        'min': Minimum value,
        'max': Maximum value,
    }
    """
    if column_name not in dataframe.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

    column_data = dataframe[column_name]
    statistics = {
        'mean': column_data.mean(),
        'median': column_data.median(),
        '25th_percentile': column_data.quantile(0.25),
        '75th_percentile': column_data.quantile(0.75),
        'min': column_data.min(),
        'max': column_data.max(),
    }
    return statistics


#%%
from prettytable import PrettyTable
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params