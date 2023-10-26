import timm
from .utils_mil import define_device
import torch
import os
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import numpy as np
import sklearn.metrics as metrics
from mil.querrybag import QuerryBag
from tqdm import tqdm

#%%
def prepare_model(model_name, sds_path):

    dev = define_device()
    # dev = "cpu"
    # model = timm.create_model('resnet152', pretrained=True, num_classes=0)

    if 'resnet152' in model_name:
        model = timm.create_model('resnet152', pretrained=True, num_classes=0)
        model.eval()
        model.to(dev)

    elif 'histoencoder' in model_name:

        from histoencoder.functional._model import create_encoder as create_encoder
        model_encoder = create_encoder("prostate_small")
        model_encoder.to(dev)
        from histoencoder.functional._features import extract_features as extract_features

        model= lambda input: extract_features(model_encoder, input)

    else:
        model_name = os.path.join(sds_path, model_name[0])
        model = models.resnet152(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 9)
        model.load_state_dict(torch.load(model_name))
        model = nn.Sequential(*list(model.children())[:-1])
        model.to(dev)

    return model


def all_elements_equal(lst):
    if not lst:  # Handling an empty list
        return True

    first_element = lst[0]

    for element in lst:
        if element != first_element:
            return False

    return True

#%%
from data.utils_data import get_slide_name
import pandas as pd
def test_loop(dl, model2test, dataset_name = None):
    correct_count = 0
    total_count = 0
    label_gt, label_pred = [], []
    bag_size = []
    if isinstance(dl, DataLoader):
        for data, bagids, labels in tqdm(dl, desc = "dataloader"):
            pred = torch.argmax(model2test((data, bagids)), dim=1)
            correct_count += (pred == labels.long()).sum()
            total_count += len(labels)
            label_gt.append(np.int64(labels.cpu().numpy()))
            label_pred.append(np.int64(pred.cpu().numpy()))
            bag_size.append(data.size())
        label_gt = np.concatenate(label_gt)
        label_pred = np.concatenate(label_pred)
        total_count = len(dl.dataset)

    elif isinstance(dl, Dataset) or isinstance(dl, QuerryBag):
        case_name = []
        for data, bagids, labels, file_list in tqdm(dl, desc = "dataset"):
            pred = torch.argmax(model2test((data, bagids)), dim=1)
            correct_count += (pred == labels.long()).sum()
            label_gt.append(np.int64(labels.cpu().numpy()))
            label_pred.append(np.int64(pred.cpu().numpy()))
            bag_size.append(data.size())
            case_name.append(get_slide_name(file_list))
        total_count = len(dl)

    acc = (correct_count / total_count) * 100
    acc = metrics.accuracy_score(label_gt, label_pred)

    if all_elements_equal(bag_size):
        print("attention, bag size is fixed")

    if dataset_name is None:
        print(f'accuracy is {acc}')
    else:
        print(f'accuracy for {dataset_name} dataset is {acc}')

    if isinstance(dl, Dataset) or isinstance(dl, QuerryBag):
        label_pred = [int(i) for i in label_pred]
        df = pd.DataFrame({'prediction': label_pred,
                   'caseID': case_name,
                    'dataset': [dataset_name] * len(label_pred)})
    else:
        df = None

    return acc, df

#%% generate random tag
import random
import string

def generate_random_tag(length=8):
    # Define the characters from which the tag will be generated
    characters = string.ascii_letters + string.digits  # You can customize this as needed

    # Generate a random tag of the specified length
    random_tag = ''.join(random.choice(characters) for _ in range(length))

    return random_tag

#%% write the tag to the yaml file
import yaml
def write_field_to_yaml(filename, tag, acc_train, acc_val, save_path, model_name, params):
    try:
        # Load existing YAML data from the file if it exists
        with open(filename, 'r') as file:
            data = yaml.safe_load(file) or {}

        model_path = os.path.join(save_path, "MIL#" + tag + "_" + model_name + ".pth")
        # Update the data with the new field
        params = clea_params(params)

        new_tag = {
            'tag': tag,
            'acc_train': acc_train,
            'acc_val': acc_val,
            'model_name': model_name,
            'save_path': save_path,
            'params': params,
            'model_type': params['model']['model_name'],
            'model_path': model_path
        }
        data[tag] = new_tag

        # Write the updated data back to the file
        with open(filename, 'w') as file:
            yaml.dump(data, file, default_flow_style=False)

        save_dict_to_file(params, os.path.join("models", "Params#" + tag + ".json"))

        print(f"Field '{tag}' successfully written to {filename}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

#%% write params to file
import json
def save_dict_to_file(dictionary, filename):
    try:
        with open(filename, 'w') as file:
            json.dump(dictionary, file, indent=4)
        print(f"Dictionary saved to {filename}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

#%% remove the odreded dicts
from collections import OrderedDict
def clea_params(params):

    params = dict(params)
    for i in params:
        t = params[i]
        if isinstance(t, OrderedDict):
            t = dict(t)
            params[i] = t

    return params