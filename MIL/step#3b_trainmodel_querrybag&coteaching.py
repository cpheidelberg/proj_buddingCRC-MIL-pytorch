
#%% save and load
import os
import platform
import hiyapyco

dataset2use = "Budding"
if dataset2use == "MINST":
    yml_file = "data/MINST/parameter_MINST.yml"
if dataset2use == "CRC":
    yml_file = "data/CRC/parameter_CRC.yml"
if dataset2use == "Budding":
    yml_file = "data/Budding/parameter_Budding.yml"

mapping = "tumor"
if mapping == "tumor":
    yml_file_spec = "data/Budding/parameter_tumor.yml"

params = hiyapyco.load(yml_file, yml_file_spec)

sds_path = params['sds_path'][platform.system()]
variables = ['ids', "labels", "instances"]

model_name = params['model']['model_name']
if "pretrainedModels" in model_name:
    model_name = "pretrained_ResNet"

from data.utils_data import get_dataset_name
save_path, model_name = get_dataset_name(model_name, dataset2use, params)

#%% load the data
from mil.utils_mil import save_variable, HDF5DataManager
HDF5 = HDF5DataManager(file_path=save_path, file_name=model_name)
instances, ids, labels, file_list = HDF5.load_datasets(dataset2load="all")
print(f"training and validation data loaded")
print(f"size of training dataset is {instances[0].size()} instances with {len(labels[0])} bags")
print(f"size of testing dataset is {instances[1].size()} instances with {len(labels[1])} bags")

#%% Initialize MilDataset using created data and mount it into a dataloader
import torch
import mil.mil as mil
from torch.utils.data import DataLoader, Subset
from mil.querrybag import QuerryBag

if params['model']['model_name'] == "histoencoder":
    normalize = True
else:
    normalize = False

bag_size_training = params['augmentation']['bag_size']
dataset_train = QuerryBag(data = instances[0], label = labels[0],
                        ids = ids[0], file_list = file_list[0],
                        normalize=normalize, bag_size =bag_size_training,
                        data_name = "datatraining") # vormals 25
dataset_train.plot_stats()
dataset_test = QuerryBag(data = instances[1], label = labels[1],
                        ids = ids[1], file_list = file_list[1],
                        normalize=normalize, bag_size =None,
                         data_name = "datavalidation")
del instances, labels, ids, file_list # to keep the memory clear

import time
t0 = time.time()
test= dataset_train[0][0]
print(test[0].size())
t1 = time.time()
print(f"time to load one dataset is {t1-t0}")
batch_size = 32
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("cpu") # formals mps
else:
    device = torch.device("cpu")

train_dl, test_dl = DataLoader(dataset_train, batch_size=batch_size, collate_fn=mil.collate,
                               drop_last=True, shuffle=True), \
                     DataLoader(dataset_test, batch_size=batch_size, collate_fn=mil.collate,
                                drop_last=True, shuffle = False)

#%% make some bag statistics
if not bag_size_training is None:
    n = 0
    for data, _, _, _ in dataset_train:
        if data.size()[0] < dataset_train.bag_size:
        #print(f"data size is {data.size()}")
            n+=1
    print(f"n = {n} bags are too small")

#%% define noise function
from utils.utils_dataaugmentation import add_row_noise, test_noise
noise2use = params["augmentation"]["noise"]
if noise2use == "poisson":
    noise_func = lambda x: add_row_noise(x,
                                noise_type="poisson",
                                noise_params={'lam':0.25})
elif noise2use == "gaussian":
    noise_func = lambda x: add_row_noise(x,
                            noise_type="gaussian",
                            noise_params={'std': 0.25, 'mean':0.5})
elif noise2use == "S&P":
    noise_func = lambda x: add_row_noise(x,
                            noise_type="S&P",
                            noise_params={'amount': 0.1, 'ratio': 0.5})
else:
    noise_func = lambda x: x

if not noise2use is None:
    test_input = dataset_train[0][0]
    #test_noise(test_input, noise_func, noise2use)
    #import matplotlib.pyplot as plt
    #plt.pause(1)
    #plt.close()
else:
    print("no error method set")

#%% model and training preparations
# model configurations
from mil.mil_model import get_model
model1 = get_model(params, dataset_train = dataset_train, train_dl = train_dl, test_dl = test_dl)
model2 = get_model(params, dataset_train = dataset_train, train_dl = train_dl, test_dl = test_dl)

# training configurations
import torch.nn as nn
n_epochs = params['model_training']['n_epochs'] #400
lr = 1e-3
criterion = nn.CrossEntropyLoss() #nn.BCEWithLogitsLoss()
optimizer1 = torch.optim.RAdam(model1.parameters(), lr=lr)
optimizer2= torch.optim.RAdam(model2.parameters(), lr=lr)

#%% --- TRAIN --- / training loop
# train on the training set
from coteaching import train, define_forget_rate
from dataclasses import dataclass
import matplotlib
vis_it = False
if vis_it:
    matplotlib.use("TkAgg")
from tqdm import tqdm
import numpy as np
forget_rate_schedule = define_forget_rate(n_epochs, forget_rate = 0.75, start_point=int(n_epochs/2))
@dataclass
class Args():
    num_iter: int
    num_iter_per_epoch: int
    epochs: int
    device: str
    print_freq: int
    batch_size: int
    forget_rate_schedule: np.ndarray
    noise_func: None

args = Args(10, 10,n_epochs, device, 250, batch_size, forget_rate_schedule, noise_func)

import matplotlib.pyplot as plt
if vis_it:
    plt.ion()
    plt.draw()

for i in tqdm(range(0, n_epochs), desc ="epoch"):

    train_acc, train_loss= train(train_dl,
        i,
        model1,
        optimizer1,
        model2,
        optimizer2,
        args)

    if vis_it:
        plt.plot(i, train_loss[0], color='green', linestyle='dashed', linewidth = 25,
             marker='o', markerfacecolor='blue', markersize=12)
        plt.draw()
        plt.pause(0.02)

    print(f"epoch #{i} loss_1 is {train_loss[0]} & loss_2 is {train_loss[1]}")

if vis_it:
    plt.ioff()
#plt.show()
print("loop closed")

#%% --- EVAL --- / validation loop
# test on training and validation set / without splitting etc.
from mil.utils_models import test_loop
import pandas as pd
model = model1
model.to("cpu")
model.eval()
model.afterNN.eval()
dataset_train.regather_cases()
dataset_train.bag_size = None
acc_train, df_train = test_loop(dataset_train, model, dataset_name="training")
acc_val, df_val = test_loop(dataset_test, model, dataset_name="validation/testing")

#%% save the datatable
import pandas as pd
import os
df = pd.concat((df_train, df_val))
from mil.utils_models import generate_random_tag
tag = generate_random_tag(10)
df.columns = [tag, "caseID", 'dataset']
model_save_name = "MIL#" + tag + "_" + model_name
df['model_name'] = tag
df_database = pd.read_excel('data/Budding/dataset.xlsx')
df_database['Nodal_binary'] = [int(i>0) for i in df_database['Nodal']]
df = df.merge(df_database, on='caseID', how ="left")

#%% save the table
file_name = os.path.join(save_path,"results.xlsx")
if os.path.exists(file_name):
    df_results = pd.read_excel(file_name)
    df = df.merge(df_results,
                  left_index=True, right_index=True,
                  how='outer', suffixes=('', '_y')
                  )
    df.drop(df.filter(regex='_y$').columns, axis=1, inplace=True)

with pd.ExcelWriter(file_name, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    df.to_excel(writer, sheet_name=tag)

with pd.ExcelWriter("models/results.xlsx", engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    df.to_excel(writer, sheet_name=tag)

#%% save the model (finally)
from mil.utils_models import write_field_to_yaml, generate_random_tag, save_dict_to_file
write_field_to_yaml("models/model_manifest.yml",
                    tag = tag,
                    acc_train=float(acc_train),
                    acc_val=float(acc_val),
                    save_path = save_path.replace(sds_path, ""),
                    model_name = model_name,
                    params = params)
torch.save(model.state_dict(), os.path.join(save_path, model_save_name+ ".pth"))
print(f"all results are saved with the tag {tag}")

