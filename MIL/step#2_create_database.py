
#%% define background
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
keys = params['mapping']['location']
if isinstance(keys, str):
    keys = [keys]

sds_path = params['sds_path'][platform.system()]

model_name = params['model']['model_name']
if "pretrainedModels" in model_name:
    model_name = "pretrained_ResNet"

from data.utils_data import get_dataset_name
save_path, model_name = get_dataset_name(model_name, dataset2use, params)

from mil.utils_mil import save_variable, HDF5DataManager
HDF5 = HDF5DataManager(file_path=save_path, file_name=model_name)
instances, ids, labels, file_list = HDF5.get_data()
print("data is loaded")

#%% look at the l-space
from lspacetools.LSpaceAnalyser import LSpaceAnalyser
l_space = instances.numpy()
vector_ids = list(ids.numpy().ravel())
l_space_label = []
vector_label = labels.numpy().tolist()
for i in range(0, l_space.shape[0]):
  l_space_label.append(int(vector_label[vector_ids[i]]))

LSpace = LSpaceAnalyser(l_space,
                        label=l_space_label,
                        show_plot=True)

if params['visualization']['plot_pca']:
    LSpace.pca(fig_title="pca-analysis" ,
                   save_path="plots/" + model_name + "_pca.png");
if params['visualization']['plot_TSNE']:
    LSpace.tsne(fig_title= "tsne-analysis",
                   save_path="plots/" + model_name + "_tsne.png");

#%% perform dataset splitting a
from utils.utils_dataaugmentation import synthetic_bags
from mil.gymbag import GymBag
from mil.gymbag import split_cases
Bag = GymBag(instances, labels, ids, file_list)
BagTrain, BagVal = Bag.__split__(0.2)
print("statistics training bag")
BagTrain.plot_stats()
bag_train = BagTrain.string_up()
bag_val = BagVal.string_up()
instances, ids, labels, file_list = [None]*2,[None]*2, [None]*2, [None]*2
instances[0] = bag_train[0]
instances[1] = bag_val[0]
labels[0] = bag_train[2]
labels[1] = bag_val[2]
ids[0] = bag_train[1]
ids[1] = bag_val[1]
file_list[0] = bag_train[3]
file_list[1] = bag_val[3]
print(f"data is split into n = {len(BagTrain)} and n = {len(BagVal)}")
from mil.utils_mil import save_database, HDF5DataManager
save_database(instances, ids, labels, file_list, save_path, suffix=model_name)

#%% perform data augmentation at the feature vector level
if not params['augmentation']['data_splitting'] is None:
    ids[0], labels[0] = split_cases(ids[0],labels[0], n_split=params['augmentation']['data_splitting'])
    print("data splitting is done")

if params['augmentation']['synthetic_data']:
    instances[0], ids[0], labels[0], file_list[0] = \
        synthetic_bags(instances[0], ids[0], labels[0], file_list[0],
                       n_steps_max=200, model = "Fast_ML")
    print("synthetic data generation is done")

print(f" after data augmentation train data set from n = {len(BagTrain)} to n = {len((labels[0]))}")

#%% save it
HDF5.save_datasets(instances, labels, ids, file_list)
print("database created and saved - everything is done")
print(f"database save to {model_name} in location {save_path} ")