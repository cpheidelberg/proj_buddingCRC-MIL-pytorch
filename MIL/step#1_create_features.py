
#%% import section
# get the files and sds_path
import pandas
import platform
import hiyapyco
import os

dataset2chose = "Budding"
if dataset2chose == "MINST":
    yml_file = "data/MINST/parameter_MINST.yml"
if dataset2chose == "CRC":
    yml_file = "data/CRC/parameter_CRC.yml"
if dataset2chose == "Budding":
    yml_file = "data/Budding/parameter_Budding.yml"

mapping = "tumor"
if mapping == "tumor":
    yml_file_spec = "data/Budding/parameter_tumor.yml"

params = hiyapyco.load(yml_file, yml_file_spec)
sds_path = params['sds_path'][platform.system()]
keys = params['mapping']['location']
if isinstance(keys, str):
    keys = [keys]

#%% load the database and adapt the files
df = pandas.read_pickle(params['images']['data_file'])
from utils.utils_image_database import adapt_file_names
if not (dataset2chose == "MINST"):
    df = adapt_file_names(df, sds_path, keys, wanna_check=False)

from matplotlib import pyplot as plt
import cv2
from random import randrange
test_file = df[keys[0]][randrange(len(df[keys[0]]))][0]
plt.imshow(cv2.imread(test_file))
plt.show()
plt.pause(1)
plt.close()

#%% generate the feature bags
from modulevaluationtools.DataLoader import FileLoader
from tqdm import tqdm
import torch
from mil.utils_mil import flatten_list, define_device, define_norm_func
import gc
from mil.utils_models import prepare_model
from utils.utils_data import create_map
model_name = params['model']['model_name']
model = prepare_model([model_name], sds_path)
from data.utils_data import get_dataset_name
save_path, model_name = get_dataset_name(model_name, dataset2chose, params)
#model.eval()
dev = define_device()
instances, ids, labels = [], [], []
file_list = []
if params['data_generation']['norm_func'] is None:
    norm_func = None
else:
    norm_func = define_norm_func(im_size=params['images']['im_size'],
                                 crop = params['data_generation']['use_image_crop'])

for i_epoch in range(0,params['data_generation']['epochs']):

    n = 0
    for i_bag in tqdm(range(0, len(df)), desc = "bag creation"):

      #%% skip empty ones
      if not df[params['mapping']['location']][i_bag]:
        continue

      #%% get the features per bag -> different bag sizes
      t_file_list = df[params['mapping']['location']][i_bag]
      file_list.append(t_file_list)
      ImageLoader = FileLoader(folder_path=t_file_list,
                               batch_size=12,
                               norm_func=norm_func,
                               im_size=params['images']['im_size']) #d
      bag = []
      for batch in tqdm(ImageLoader, desc ="batch"):
        bag.append(model(batch.to(dev)).detach().to('cpu'))

      bag = torch.cat(bag, dim = 0)
      instances.append(bag)

      #%% get one nice image
      if i_bag == 0:
        batch = ImageLoader[0]
        create_map(batch, save_path=os.path.join(save_path, model_name + "_1st_batch_epoch#" + str(i_epoch)))

      #%% get the bag ids
      ids.append([n] * len(ImageLoader.image_file_list))
      n+=1
      # get the label
      labels.append([df[params['mapping']['label']][i_bag]])

      #%% clean up
      del ImageLoader #, model
      gc.collect()
      torch.cuda.empty_cache()

ids = torch.tensor(flatten_list(ids))
labels = torch.tensor(flatten_list(labels))
instances = torch.cat(instances, dim =0)
file_list = flatten_list(file_list)

#%% save and load
from mil.utils_mil import save_variable, HDF5DataManager
#save_variable(variable_name="ids", variable=ids, save_folder=save_path, suffix=model_name);
#save_variable(variable_name="labels", variable=labels, save_folder=save_path, suffix=model_name);
#save_variable(variable_name="instances", variable=instances, save_folder=save_path, suffix=model_name);
#save_variable(variable_name="file_list", variable=file_list, save_folder=save_path, suffix=model_name);

HDF5 = HDF5DataManager(file_path=save_path, file_name=model_name)
HDF5.save_data(instances, labels, ids, file_list)
print(f"file {model_name} has been saved in {save_path}")
print("saving finished")
