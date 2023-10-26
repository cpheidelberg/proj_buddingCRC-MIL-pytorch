import torch
from mil.bags import Bag
import random
import numpy as np
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

#%% define the GymBag class
class GymBag():
    def __init__(self, data=[], label= [], ids = None, file_list = []):
        #super().__init__(data, ids, label, file_list)

        self.data = data
        self.ids = ids
        self.file_list = file_list
        self.bags = torch.unique(self.ids)
        if len(label) == len(self.bags):
            self.label = label
        else:
            self.label = label[0:np.int64(torch.max(self.bags))]
            print("caution numbers labels and bags are discrepant")
    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data = self.data[self.ids == self.bags[index]]
        bagids = torch.ones(data.size()[0]) * self.bags[index]
        labels = self.label[index]

        if not self.file_list is None:
            index = self.ids == self.bags[index]
            index = index.tolist()
            file_list = [i for (i, v) in zip(self.file_list, index) if v]
        else:
            file_list = None

        return data, bagids, labels, file_list

    def __split__(self, split_factor):

        instances_train, bagids_train, labels_train, file_list_train = [], [], [], []
        instances_val, bagids_val, labels_val, file_list_val = [], [], [], []
        for i in range(0, len(self.label)):

            i_instances, i_bagids, i_labels, i_file_list = self.__getitem__(i)
            if flip_coin(split_factor):
                instances_val.append(i_instances)
                bagids_val.append(i_bagids)
                labels_val.append(i_labels)
                file_list_val.append(i_file_list)
            else:
                instances_train.append(i_instances)
                bagids_train.append(i_bagids)
                labels_train.append(i_labels)
                file_list_train.append(i_file_list)

        instances_train = torch.cat(instances_train, dim =0 )
        bagids_train = torch.cat(bagids_train, dim=0)
        labels_train = torch.tensor(labels_train)
        file_list_train = unnest_list(file_list_train)
        BagTrain = Bag(instances_train, labels_train, bagids_train, file_list_train, use_data_dump=False)
        instances_val = torch.cat(instances_val, dim=0)
        bagids_val = torch.cat(bagids_val, dim=0)
        labels_val = torch.tensor(labels_val)
        file_list_val = unnest_list(file_list_val)
        BagVal = Bag(instances_val, labels_val, bagids_val, file_list_val, use_data_dump=False)
        self.BagTrain = BagTrain
        self.BagVal = BagVal

        return BagTrain, BagVal

#%% split it
import numpy as np
import random
from tqdm import tqdm
def split_cases(ids, labels_in, n_split = 4):

    ids_out = ids.clone()

    if len(labels_in) != len(np.unique(ids)):
        labels_in = labels_in[0:len(np.unique(ids))]

    n = 0
    labels_out = []
    for i_ids in tqdm(np.unique(ids),desc="case"):
        idx = np.where(ids == i_ids)[0].tolist()
        values = list(range(0, n_split))
        values = [i + (n * n_split) for i in values]

        #print(values)
        for i in idx:
            ids_out[i] = random.choice(values) + 10e6
        labels_out.append([labels_in[n]] * n_split)
        n += 1

    ids_out = [i-10e6 for i in ids_out]
    ids_out = torch.stack(ids_out)
    labels_out = [item for sublist in labels_out for item in sublist]
    labels_out = torch.stack(labels_out)

    return ids_out, labels_out
