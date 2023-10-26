#%% import section
import torch
import torch.nn.functional as F
from mil.utils_mil import flatten_list
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import random
from skimage.util import random_noise

#%% add row-wise error
def add_row_noise(tensor, noise_type, noise_params):
    """
    Adds row-wise noise to a PyTorch tensor.

    Args:
        tensor (torch.Tensor): The input tensor.
        noise_type (str): The type of noise to add ('gaussian' or 'poisson').
        noise_params (dict): Parameters specific to the noise type.
                             For Gaussian noise, it should contain 'std' for standard deviation.
                             For Poisson noise, it should contain 'lam' for lambda value.

    Returns:
        torch.Tensor: The tensor with row-wise noise added.
    """
    if noise_type == 'gaussian':
        std_value = noise_params['std']
        mean_value = noise_params['mean']
        noise = torch.tensor(np.random.normal(mean_value, std_value, tensor.size()), dtype=torch.float)

    elif noise_type == 'poisson':
        lam = noise_params['lam']
        noise = []
        for i in range(tensor.size()[0]):
            noise.append(torch.poisson(torch.ones(1, tensor.size()[1]) * lam).squeeze())
        noise = torch.stack(noise, dim = 0)

    elif noise_type == "S&P":
        noisy_tensor = random_noise(tensor, mode="S&P", amount=noise_params["amount"])
        noisy_tensor = torch.from_numpy(noisy_tensor)
        return noisy_tensor

    noisy_tensor = tensor  - (tensor * noise.float())

    return noisy_tensor

def test_noise(data, noise_func, noise2use):
    test_input = data
    test_output = noise_func(test_input)
    print(f"The following noise-generation for data augmentation is used {noise2use}")
    difference_map = test_input - test_output
    sum_diff = mean_squared_error(difference_map, data)
    if sum_diff == 0:
        print("warning, no error added at all")
    else:
        print(f"mean squared error is {sum_diff}")

    plt.subplot(211)
    plt.imshow(test_input.numpy())
    plt.colorbar()
    plt.title("original")
    plt.subplot(212)
    plt.imshow(test_output.numpy())
    plt.colorbar()
    plt.title("noisy")
    plt.tight_layout()
    plt.show()

#%% helper functions
import pandas as pd

def matrix2dataframe(matrix, prefix):

    dataframe = pd.DataFrame(matrix)
    new_columns = []
    n_cols = len(dataframe.columns)

    # Generate new column names
    for i in range(n_cols):
        new_name = prefix + str(i)
        new_columns.append(new_name)

    # Assign new column names to the DataFrame
    dataframe.columns = new_columns

    return dataframe

def dataframe2matrix(dataframe):

    matrix = dataframe.values
    matrix = torch.from_numpy(matrix)

    return matrix

#%% create synthetic bags
import numpy as np
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.lite import SingleTablePreset
import random
from tqdm import tqdm

def synthetic_bags(instances, ids, labels, file_list, n_steps_max = None, model = "Fast_ML"):

    n_bags = np.unique(ids.numpy())
    np.random.shuffle(n_bags)
    instances_synthetic, ids_synthetic, labels_synthetic, file_list_synthetic = [],[],[],[]

    if n_steps_max is None:
        n_steps_max = len(n_bags)

    n_steps, i_step = 0,0
    real_step = 0
    while n_steps < n_steps_max:
        i_bag = int(n_bags[i_step])
        bag_instances = instances.numpy()[ids==i_bag]
        bag_instances = matrix2dataframe(bag_instances, "feature")

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(bag_instances)
        if model == "Fast_ML":
            synthesizer = SingleTablePreset(metadata, name='FAST_ML')
        elif model == "GaussianCopula":
            synthesizer = GaussianCopulaSynthesizer(metadata)
        synthesizer.fit(data=bag_instances)
        bag_instances_synthetic = synthesizer.sample(num_rows=len(bag_instances))
        instances_synthetic.append(dataframe2matrix(bag_instances_synthetic))

        ids_synthetic.append([max(n_bags)+ i_bag +1] * len(bag_instances_synthetic))
        labels_synthetic.append(labels[i_bag])
        bag_file_list = [element for element, is_selected in zip(file_list, ids.squeeze()==i_bag) if is_selected]
        file_list_synthetic.append(bag_file_list)

        print(f"synthetic data creation step {real_step}")
        real_step +=1
        n_steps += 1
        i_step += 1
        if i_step >= len(n_bags):
            i_step = 0

    ids_synthetic = torch.tensor(flatten_list(ids_synthetic))
    labels_synthetic = torch.tensor(flatten_list(labels_synthetic))
    instances_synthetic = torch.cat(instances_synthetic, dim=0)
    file_list_synthetic = flatten_list(file_list_synthetic)

    ids_synthetic = torch.cat((ids, ids_synthetic), dim=0)
    labels_synthetic = torch.cat((labels, labels_synthetic), dim = 0)
    instances_synthetic = torch.cat((instances, instances_synthetic), dim=0)
    file_list_synthetic = file_list + file_list_synthetic

    return instances_synthetic, ids_synthetic, labels_synthetic, file_list_synthetic






