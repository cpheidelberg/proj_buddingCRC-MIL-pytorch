
import torch
import numpy as np
import random
#from mil.bagsselective import BagSelective, BagNegative, BagPositive
import random
import h5py
from tqdm import tqdm

#%% define the bag class
import os
import re
def get_case_name(file_name):

    t_file_name = os.path.basename(file_name)
    t_file_name = t_file_name[0:t_file_name.find("_HE")]
    t_file_name = re.findall(r'\d+', t_file_name)
    case_name = "CaseID#" + t_file_name[0]
    return case_name

#%%
def matrix_to_list(matrix):

    if isinstance(matrix, str):
        return matrix

    result = []
    for row in matrix:
        result.append(row)
    return result

def flip_coin(probability):
    """
    Returns True with a given probability, otherwise returns False.

    :param probability: The probability of returning True (a float between 0 and 1).
    :return: True with the specified probability, False otherwise.
    """
    if not 0 <= probability <= 1:
        raise ValueError("Probability must be a float between 0 and 1.")

    return random.random() < probability

def unnest_list(nested_list):
    result = []

    for item in nested_list:
        if isinstance(item, list):
            result.extend(unnest_list(item))
        else:
            result.append(item)

    return result

def count_repeated_values(tensor):
    """
    Count the number of repeated values in a tensor (NumPy array).

    Args:
    tensor (numpy.ndarray): The input tensor.

    Returns:
    dict: A dictionary where keys are unique values in the tensor, and values are the counts of each unique value.
    """
    unique_values, counts = np.unique(tensor, return_counts=True)
    result = dict(zip(unique_values, counts))
    return result

class Bag:
    def __init__(self, data=[], label= [], ids = None, file_list = [], weight = None, data_name = None, use_data_dump = True):
        #super().__init__(data, ids, label, file_list, normalize)

        if isinstance(data, Bag):
            label = data.label
            ids = data.ids.squeeze()
            file_list = data.file_list
            weight = data.weight
            data = data.data

        if ids is None:
            if len(data)>0:
                self.ids = torch.zeros(data.size()[0])
            else:
                self.ids = None
        else:
            self.ids = ids.squeeze()

        self.bags = torch.unique(self.ids)

        self.data_sizeY, self.data_sizeX = data.size()
        self.data_name = data_name

        if data.size()[0] > 1e3 and use_data_dump:
            if data_name is None:
                file_name = "data_dump.h5"
            else:
                file_name = "data_dump_" + data_name +".h5"
            h5f = h5py.File(file_name, 'w')
            h5f.create_group('data')
            h5f.create_group('bagids')
            h5f_data = h5f['data']
            h5f_bagids = h5f['bagids']
            for i, v in tqdm(enumerate(np.unique(self.ids)), desc = "data dumping"):
                t_data = data[self.ids == v]
                h5f_data.create_dataset(str(int(v)), data=t_data)
                t_bagids = self.ids[self.ids.squeeze() == self.bags[int(i)]]
                h5f_bagids.create_dataset(str(int(v)), data = t_bagids)
            h5f.close()
            data = file_name
            print("data dumping used")
        self.data = data

        self.label = label
        if len(data)>0:
            self.instance_list = matrix_to_list(self.data)
        else:
            self.instance_list = []
        self.weight = weight
        self.file_list = file_list

        if len(data)>0:
            self._data_ = matrix_to_list(self.data)
        else:
            self._data_ = []

        self._ZnMix_ = False
        self.use_data_dump = use_data_dump

    def append(self, data, label, ids = None, file_list = [], weight = None):

        if len(self.data) >0 and len(self.label) > 0:
            self.data = torch.cat((self.data, data), dim=0)
            self.label = torch.cat((self.label, label), dim=0)

            if ids is None:
                if self.ids is None:
                    self.ids = torch.ones(data.size()[0]) * 0
                else:
                    self.ids = torch.cat((self.ids, torch.ones(data.size()[0]) * (torch.max(self.ids) + 1)),dim=0)
            else:
                #ids = ids - torch.min(ids)
                self.ids = torch.cat((self.ids, ids.squeeze()),dim=0)

            if len(self.file_list) > 0 and len(file_list) > 0:
                self.file_list.append(file_list)
            else:
                self.file_list = []

            if not self.weight is None and not weight is None:
                self.weight = np.concatenate((self.weight, weight), axis=0)
            else:
                self.weight = None

        else:
            self.data = data
            self.label = label
            self.ids = ids.squeeze()
            self.file_list = file_list
            self.weight=weight

    def __add__(self, other):

        if isinstance(other, Bag) or isinstance(other, BagSelective):

            data = torch.cat((self.data, other.data), dim=0)
            instance_list = matrix_to_list(self.data)

            if not self.weight is None and not other.weight is None:
                weight = np.concatenate((self.weight, other.weight), axis =0)
            else:
                weight = None

            label = torch.cat((self.label, other.label), dim=0)

            ids = other.ids - torch.min(other.ids)
            ids = ids + torch.max(self.ids) + 1
            ids = torch.cat((self.ids, ids), dim = 0)

            file_list = self.file_list.append(other.file_list)

            if file_list is None:
                file_list = []

            if not len(file_list) == data.size()[0]:
                file_list = []

            return Bag(data=data, ids=ids, label=label, file_list=file_list, weight=weight)

    def plot_stats(self):

        n_bags =len(np.unique(self.ids))
        print(f"{n_bags} bags are included")
        if torch.is_tensor(self.data):
            n_instances = self.data.size()[0]
        elif isinstance(self.data, str):
            n_instances = self.data_sizeY
        else:
            n_instances = self.data.shape[0]

        print(f"{n_instances} instances in total")
        print(f"Mean {n_instances/n_bags} instances per bag")
        print(f"{len(self.file_list)} files are linked to the bags / the bag container")

        for i in list(np.unique(self.label)):
            print(f"{100 * torch.sum(self.label == i)/n_bags}% label {i} included")

        if self.weight is not None:
            print(f"Mean weight is: {np.mean(self.weight)} with +/- {np.std(self.weight)} "
                  f"and min/max {np.min(self.weight)} and {np.max(self.weight)}")

    def string_up(self):
        return self.data, self.ids, self.label, self.file_list

    def __len__(self):
        if hasattr(self, 'bags'):
            return len(self.bags)
        elif hasattr(self, 'label'):
            return len(self.label)
        else:
            print("not defined")
            return None

    def remix_it(self, num_sublists=None):

        list2mix = matrix_to_list(self.data)
        mix = list(range(0, len(list2mix)))
        random.shuffle(mix)

        def shuffle_list(list_input):
            list_output = []
            for i in range(0, len(list_input)):
                list_output.append(list_input[mix[i]])
            return list_output

        list2mix = shuffle_list(list2mix)
        file_list = self.file_list

        if len(file_list) > 0:
            file_list = unnest_list(file_list)
            file_list = shuffle_list(file_list)

        if num_sublists is None:
            _, counts = np.unique(self.ids, return_counts=True)
            sublist_size = round(np.mean(counts))
            num_sublists = len(list2mix) // sublist_size
            remainder = len(list2mix) % num_sublists

        def mix_function(list_input):
            sublists = []
            for i in range(0, len(list_input) - sublist_size, sublist_size):
                start_index = i
                stop_index = i + sublist_size
                sublist = list_input[start_index:stop_index]
                if isinstance(sublist[0], str):
                    sublists.append(sublist)
                else:
                    sublists.append(torch.stack(sublist))

            if remainder > 0:
                start_index = stop_index
                stop_index = start_index + remainder
                sublist = list_input[start_index:stop_index]
                if isinstance(sublist[0], str):
                    sublists.append(sublist)
                else:
                    sublists.append(torch.stack(list_input[start_index:stop_index]))

            return sublists

        list2mix = mix_function(list2mix)
        data = list2mix[0]
        for i in range(1, len(list2mix)):
            data = torch.cat((data, list2mix[i]), dim=0)
        self.data = data

        if len(file_list) > 0:
            file_list = mix_function(file_list)
        self.file_list = unnest_list(file_list)

        ids = torch.ones(list2mix[0].size()[0]) * 0
        for i in range(1, len(list2mix)):
            ids = torch.cat((ids, torch.ones(list2mix[0].size()[0]) * i))
        self.ids = ids

        if np.unique(self.label) == 0:
            self.label = torch.zeros(len(np.unique(ids)))
        else:
            self.label = torch.ones(len(np.unique(ids)))
        self._ZnMix_ = True

    def regather_cases(self):

        if self.use_data_dump:
            data = []
            h5f = h5py.File(self.data, 'r')
            for index2load in tqdm(self.bags, desc="regather data"):
                t_data = h5f['data'][str(int(index2load))]
                data.append(torch.tensor(np.array(t_data)))
            h5f.close()
            data = torch.cat(data)

        case_list = [get_case_name(i_file) for i_file in self.file_list]
        unique_caseID = np.unique(case_list)

        ids = self.ids
        label_per_ids = torch.zeros(ids.size())
        for i, v in enumerate(self.bags):
            label_per_ids[ids == v] = self.label[i]

        label = []
        for idx, caseID in tqdm(enumerate(unique_caseID), desc="regathering ids"):
            index2load = [i == caseID for i in case_list]
            ids[index2load] = idx
            print(1)
            t_label = label_per_ids[index2load]
            print("2")
            label.append(torch.mode(t_label)[0])
            print(label)
        print(3)
        #label = torch.cat(label)
        print(label)

        # set the new bags
        print("new values set")
        self.bags = torch.unique(ids)  # set the new bags
        self.label = label  # set the new label
        self.ids = ids

        if self.use_data_dump:
            h5f = h5py.File(self.data, 'w')
            h5f.create_group('data')
            h5f.create_group('bagids')
            h5f_data = h5f['data']
            h5f_bagids = h5f['bagids']
            for i, v in tqdm(enumerate(np.unique(self.ids)), desc = "data re-dumping"):
                #print(i)
                t_data = data[self.ids == v,:]
                h5f_data.create_dataset(str(int(v)), data=t_data)
                t_bagids = self.ids[self.ids.squeeze() == v]
                h5f_bagids.create_dataset(str(int(v)), data = t_bagids)
            h5f.close()

        print("all new set")


