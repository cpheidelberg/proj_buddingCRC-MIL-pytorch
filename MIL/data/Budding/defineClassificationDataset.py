

#%% import section
import os
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
import platform
import pandas as pd
import yaml

tag ="xrmATxjNy0"
# tag fKspnHMSMb is the best model for n_neuron = 15 and 10epoch-data by histoencoder
#tag = "fKspnHMSMb"
# get the corresponding cases from the database

with open("models/model_manifest.yml", 'r') as yaml_file:
    model_manifest = yaml.safe_load(yaml_file)

model_manifest = model_manifest[tag]
params = model_manifest['params']
model_name = model_manifest['model_name']
sds_path= params['sds_path'][platform.system()]
save_path = os.path.join(sds_path, model_manifest['save_path'])
xlsx_file_name = os.path.join(os.path.dirname(save_path), "featureVector_df_TAG#" + tag + ".xlsx")
df_database = pd.read_excel(xlsx_file_name, sheet_name=tag + "_probabilities")

#%%
df_database = df_database.groupby('file').agg({'caseID': ' '.join,
                                               'dataset': ' '.join,
                                               'prob': 'mean',
                                               'mse': 'mean',
                                               'image': ' '.join,
                                               'location': ' '.join}).reset_index()
print("files are loaded")

#%% adapt the database
norm_func = lambda x: (x - 0) / (1 - 0) * 2 - 1
df_database['label_prob'] = df_database.prob.apply(norm_func)

def class_decision(i):
    if i > 0.75:
        c = 2
    elif i < -0.75:
        c = 1
    else:
        c =0
    return c

df_database['image_class'] = df_database.label_prob.apply(class_decision)
#for i in range(0,3):
#    print(f'n = {np.sum(df_database.image_class==i) / len(df_database)} examples for class #{i}')

#%% prepare the validation and training dataset (and dump it to hdf5)
from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df_database, test_size=0.2)
df_train.to_pickle(os.path.join(os.path.dirname(os.path.dirname(save_path)), "df_train.pkl"))
df_test.to_pickle(os.path.join(os.path.dirname(os.path.dirname(save_path)), "df_test.pkl"))

#%% define the hdf5-saving file
import os
import h5py
from PIL import Image
from torchvision import transforms
img_size = 512
def images_to_hdf5(image_list, hdf5_filename):
    # Get a list of image file paths in the specified directory
    def remove_sds_path(file_name):
        idx = file_name.find("sd18a006")
        return file_name[idx:]

    # Create an HDF5 file for saving the images
    print(f"n={len(image_list)} images are dumped to {hdf5_filename}")

    with h5py.File(hdf5_filename, 'w') as hdf5_file:
        # Create a dataset to store the images as tensors
        image_dataset = hdf5_file.create_dataset("images", (len(image_list), 3, img_size, img_size), dtype='f')

        # Define a transformation to preprocess the images (resize, normalize, etc.)
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load and preprocess each image, then save it in the HDF5 file
        for i, image_file in enumerate(tqdm(image_list, desc = "images dumped to hdf5")):
            image_file = os.path.join(sds_path, remove_sds_path(image_file))
            image = Image.open(image_file)
            image = transform(image)
            image_dataset[i] = image.numpy()

df_train = pd.read_pickle(os.path.join(os.path.dirname(os.path.dirname(save_path)), "df_train.pkl"))
hdf5_train = os.path.join(os.path.dirname(os.path.dirname(save_path)), "hdf5_train_dump.h5")
#hdf5_train = os.path.basename(hdf5_train)
images_to_hdf5(list(df_train.file), hdf5_train)
df_test = pd.read_pickle(os.path.join(os.path.dirname(os.path.dirname(save_path)), "df_test.pkl"))
hdf5_val = os.path.join(os.path.dirname(os.path.dirname(save_path)), "hdf5_val_dump.h5")
images_to_hdf5(list(df_test.file), hdf5_val)

print("all data belongs to hdf5 ;) ")