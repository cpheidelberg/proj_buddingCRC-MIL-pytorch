import torch
import h5py
from tqdm import tqdm

#%% flattern list
def flatten_list(nested_list):
  flattened = []
  for item in nested_list:
    if isinstance(item, list):
      flattened.extend(flatten_list(item))
    else:
      flattened.append(item)
  return flattened

def define_device():
    if torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cpu"
    return dev

#%% modify the file name
def add_suffix_to_filepath(filepath, suffix):
    path, filename = os.path.split(filepath)
    filename, ext = os.path.splitext(filename)
    new_filename = f"{filename}_{suffix}{ext}"
    new_filepath = os.path.join(path, new_filename)
    return new_filepath

#%% save the results
import pickle
import os
import torch

def save_variable(variable_name, variable, save_folder = "data/", suffix = None):
    filename = os.path.join(save_folder, (variable_name + ".pt"))

    if not (suffix is None):
        filename = add_suffix_to_filepath(filename, suffix)

    if os.path.exists(filename):
        os.remove(filename)

    torch.save(variable, filename)
    print(f"Variable '{variable}' saved to file '{filename}'.")

def load_variable(variable, save_folder = "data/", suffix = None):
    filename = os.path.join(save_folder, (variable + ".pt"))

    if not (suffix is None):
        filename = add_suffix_to_filepath(filename, suffix)

    if os.path.isfile(filename):
        # Variable doesn't exist in the workspace, but the file exists, load it
        with open(filename, 'rb') as file:
            t_variable = torch.load(file)
            globals()[variable] = t_variable
        print(f"Variable '{variable}' loaded from file '{filename}'.")
        return t_variable

    else:
        print(f"For variable '{variable}' the file '{filename}' does not exist.")

#%%
from torchvision import transforms
def define_norm_func(im_size, crop = False):

    if crop:
        eval_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop((im_size,im_size)),
            transforms.Resize((im_size, im_size)),
            transforms.RandomRotation(degrees=365),
            transforms.RandomGrayscale(),
            transforms.RandomAdjustSharpness(sharpness_factor=0.1),
            transforms.Resize((im_size, im_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
    else:
        eval_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((im_size, im_size)),
            transforms.RandomRotation(degrees=365),
            transforms.RandomGrayscale(),
            transforms.RandomAdjustSharpness(sharpness_factor=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

    return eval_transforms

#%% load and save database
def save_database(instances, ids, labels, file_list, save_folder = "data/", suffix = None):

    database = {'instances': instances, "ids": ids, "labels": labels, "file_list": file_list}

    filename = os.path.join(save_folder,  "database.pt")

    if not (suffix is None):
        filename = add_suffix_to_filepath(filename, suffix)

    with open(filename, 'wb') as f:
       torch.save(database, f)

def load_database(save_folder = "data/", suffix = None):

    filename = os.path.join(save_folder, "database.pt")

    if not (suffix is None):
        filename = add_suffix_to_filepath(filename, suffix)

    if os.path.isfile(filename):
        # Variable doesn't exist in the workspace, but the file exists, load it
        with open(filename, 'rb') as file:
            database = torch.load(file)

        print(f"Database loaded from file '{filename}'.")

        instances = database['instances']
        ids = database['ids']
        labels = database['labels']
        file_list = database['file_list']
        return instances, ids, labels, file_list

    else:
        print(f"For the database the file '{filename}' does not exist.")

#%%
import torch

def shuffle_rows(tensor):
    # Get the number of rows in the tensor
    num_rows = tensor.size(0)

    # Generate a random permutation of row indices
    shuffled_indices = torch.randperm(num_rows)

    # Use the permutation to shuffle the rows
    shuffled_tensor = tensor[shuffled_indices]

    return shuffled_tensor


#%%
import numpy as np
class HDF5DataManager:
    def __init__(self, file_path, file_name):
        self.file_path = file_path
        self.file_name = file_name
        self.instances = []
        self.labels = []
        self.ids = []
        self.file_list = []

    def save_data(self, instances, labels, ids, file_list):
        file_name = os.path.join(self.file_path, self.file_name) + ".h5"
        hf = h5py.File(file_name, 'w')
        hf.create_dataset('instances', data=instances)
        hf.create_dataset('labels', data=labels)
        hf.create_dataset("ids", data = ids)
        hf.create_dataset("file_list", data = file_list)
        hf.close()
        self.instances = instances
        self.labels= labels
        self.ids = ids
        self.file_list = file_list

    def get_data(self):
        file_name = os.path.join(self.file_path, self.file_name) + ".h5"
        hf = h5py.File(file_name, 'r')

        keys = ['instances', 'ids', 'labels']
        output = []
        for i_key in tqdm(keys, desc = "data loading"):
            output.append(torch.from_numpy(np.array(hf.get(i_key))))

        instances = output[0]
        ids = output[1]
        labels = output[2]

        file_list = list(hf.get("file_list"))
        file_list = [i.decode() for i in file_list]
        hf.close

        self.instances = instances
        self.labels = labels
        self.ids = ids
        self.file_list = file_list

        return self.instances, self.ids, self.labels, self.file_list

    def save_datasets(self, instances, labels, ids, file_list):
        file_name = os.path.join(self.file_path, self.file_name) + ".h5"
        hf = h5py.File(file_name, 'a')

        try:
            training_data = hf.create_group("training_data")
        except:
            del hf['training_data']
            training_data = hf.create_group("training_data")

        training_data.create_dataset("instances", data = instances[0])
        training_data.create_dataset("labels", data=labels[0])
        training_data.create_dataset("ids", data=ids[0])
        training_data.create_dataset("file_list", data=file_list[0])

        try:
            validation_data = hf.create_group("validation_data")
        except:
            del hf['validation_data']
            validation_data = hf.create_group("validation_data")
        validation_data.create_dataset("instances", data = instances[1])
        validation_data.create_dataset("labels", data=labels[1])
        validation_data.create_dataset("ids", data=ids[1])
        validation_data.create_dataset("file_list", data=file_list[1])
        hf.close()

    def load_datasets(self, dataset2load = "all"):
        file_name = os.path.join(self.file_path, self.file_name) + ".h5"
        output = []
        keys = ['instances', 'ids', 'labels']
        hf = h5py.File(file_name, 'r')

        if dataset2load == "training" or dataset2load == "all":
            for i_key in tqdm(keys, desc = "loading training data"):
                output.append(torch.from_numpy(np.array(hf['training_data'].get(i_key))))
            file_list = list(hf['training_data'].get("file_list"))
            output.append([i.decode() for i in file_list])
        if dataset2load == "validation" or dataset2load == "all":
            for i_key in tqdm(keys, desc="loading validation data"):
                output.append(torch.from_numpy(np.array(hf['validation_data'].get(i_key))))
            file_list = list(hf['validation_data'].get("file_list"))
            output.append([i.decode() for i in file_list])

        if dataset2load == "training" or dataset2load == "validation":
            instances = output[0]
            ids = output[1]
            labels = output[2]
            file_list = output[3]
        else:
            instances, ids, labels, file_list=[],[],[],[]
            for i in range(0,2):
                instances.append(output[0 + 4 *i])
                ids.append(output[1+ 4 *i])
                labels.append(output[2+ 4 *i])
                file_list.append(output[3 + 4 *i])
        hf.close()
        self.instances = instances
        self.labels = labels
        self.ids = ids
        self.file_list = file_list

        return self.instances, self.ids, self.labels, self.file_list

