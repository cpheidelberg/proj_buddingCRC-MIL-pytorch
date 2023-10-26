
import torch
import numpy as np
import random
from mil.bags import Bag

#%% define the selective Bag
def unique_orderd(arr):
    # Applying unique function on array
    res, ind = np.unique(arr, return_index=True)

    # Sorting indices
    result = res[np.argsort(ind)]

    return result

def matrix_to_list(matrix):
    result = []
    for row in matrix:
        result.append(row)
    return result

class BagSelective(Bag):
    def __init__(self, data=[], label=[], ids = None, label2include = None, file_list = [], weight = None):
        super().__init__(data, label, ids, file_list, weight)

        if (isinstance(data, Bag) or isinstance(data, BagSelective)
                or isinstance(data, BagPositive) or isinstance(data, BagNegative)):
            label = data.label
            ids = data.ids
            file_list = data.file_list
            weight = data.weight
            data = data.data

        self.label2include = label2include

        if not (label2include is None):
            id2include = torch.where(label == label2include)[0]
            self.label = label[id2include]

            id = torch.zeros(len(ids))
            ids_list = unique_orderd(ids)
            for i in range(0, len(id2include)):
                id += ids == ids_list[id2include[i]]
            id2include = id.bool()
            self._id2include_ = id2include

            self.data = data[id2include]

            if ids is None:
                self.ids = torch.zeros(data.size()[0])
            else:
                self.ids = ids[id2include]

            if  len(file_list) > 0 and len(file_list) == data.size()[0]:
                self.file_list = [file_list[i] for i in range(len(id2include)) if id2include[i]]

            if not weight is None:
                self.weight = weight[id2include]
            else:
                self.weight = weight

    def append(self, data, label, ids=None, file_list=[], weight=None):

        if self.label2include is None:

            self.data = torch.cat((self.data, data), dim=0)
            self.label = torch.cat((self.label, label), dim=0)

            if ids is None:
                self.ids = torch.cat((self.ids, torch.ones(data.size()[0] * (torch.max(self.ids) + 1))), dim=0)
            else:
                ids = ids - torch.min(ids)
                self.ids = ids + torch.max(self.ids) + 1

            if len(self.file_list) >0  and len(file_list) >0:
                self.file_list.append(file_list)
            else:
                self.file_list = []

            if not self.weight is None and not weight is None:
                self.weight = np.cat((self.weight, weight), dim=0)
            else:
                self.weight = None

        else:
            #print('test')
            id2include = torch.where(label == self.label2include)[0]
            self.label = torch.cat((self.label, label[id2include]), dim=0)

            id = torch.zeros(len(ids))
            ids_list = unique_orderd(ids)
            for i in range(0, len(id2include)):
                id += ids == ids_list[id2include[i]]
            id2include = id.bool()

            self.data = torch.cat((self.data, data[id2include]), dim = 0)

            if ids is None:
                self.ids = torch.zeros(data.size()[0])
            else:
                self.ids = torch.cat((self.ids, ids[id2include]), dim = 0)

            if len(self.file_list) > 0:
                self.file_list.append([file_list[i] for i in range(len(id2include)) if id2include[i]])

            if not self.weight is None and not weight is None:
                self.weight = np.cat((self.weight, weight[id2include]), dim=0)
            else:
                self.weight = None

#%% define the Bag-Class for negative cases
def unnest_list(nested_list):
    result = []

    for item in nested_list:
        if isinstance(item, list):
            result.extend(unnest_list(item))
        else:
            result.append(item)

    return result

class BagNegative(BagSelective):
    def __init__(self, data, label=None, ids = None, file_list = [], weight = None):
        super().__init__(data, label, ids = ids, label2include = 0, file_list = file_list, weight = weight)

        self._ZnMix_ = False
    def remix_it(self, num_sublists = None):

        list2mix = matrix_to_list(self.data)
        mix = list(range(0, len(list2mix)))
        random.shuffle(mix)

        def shuffle_list(list_input):
            list_output = []
            for i in range(0, len(list_input)):
                list_output.append(list_input[mix[i]])
            return list_output

        list2mix =shuffle_list(list2mix)
        file_list = self.file_list

        if len(file_list) >0:
            file_list = unnest_list(file_list)
            file_list = shuffle_list(file_list)

        if num_sublists is None:
            _, counts = np.unique(self.ids, return_counts=True)
            sublist_size = round(np.mean(counts))
            num_sublists = len(list2mix) // sublist_size
            remainder = len(list2mix) % num_sublists

        def mix_function(list_input):
            sublists = []
            for i in range(0, len(list_input)-sublist_size, sublist_size):
                start_index = i
                stop_index = i + sublist_size
                sublist = list_input[start_index:stop_index]
                if isinstance(sublist[0], str):
                    sublists.append(sublist)
                else:
                    sublists.append(torch.stack(sublist))

            if remainder > 0:
                start_index = stop_index
                stop_index = start_index +remainder
                sublist = list_input[start_index:stop_index]
                if isinstance(sublist[0], str):
                    sublists.append(sublist)
                else:
                    sublists.append(torch.stack(list_input[start_index:stop_index]))

            return sublists

        list2mix = mix_function(list2mix)
        data = list2mix[0]
        for i in range(1, len(list2mix)):
            data=torch.cat((data, list2mix[i]), dim = 0)
        self.data = data

        if len(file_list) > 0:
            file_list = mix_function(file_list)
        self.file_list = unnest_list(file_list)

        ids = torch.ones(list2mix[0].size()[0]) * 0
        for i in range(1, len(list2mix)):
            ids = torch.cat((ids, torch.ones(list2mix[0].size()[0]) * i))
        self.ids = ids

        self.label = torch.zeros(len(np.unique(ids)))
        self._ZnMix_ = True

#%% define the Bag-Class for positive cases
import random

def pick_n_random_elements(input_list, n):

    if n >= len(input_list):
        # If n is greater than or equal to the list length, return the whole list
        result_list = input_list.copy()
        input_list = []

    else:
        idx = random.sample(range(0, len(input_list)), n)
        result_list = [input_list[i] for i in idx]

        for i in sorted(idx, reverse=True):
            input_list.pop(i)

    return result_list, input_list

def mean_length_of_elements(input_list):
    total_length = 0
    for element in input_list:
        total_length += len(element)

    return total_length / len(input_list)

def are_all_elements_none(input_list):
    return all(element is None for element in input_list)

def mix_bags_function(list_input, sublist_size):

    num_sublists = len(list_input) // sublist_size
    remainder = len(list_input) % num_sublists

    sublists = []
    for i in range(0, len(list_input) - sublist_size, sublist_size):
        start_index = i
        stop_index = i + sublist_size
        sublist = list_input[start_index:stop_index]
        sublists.append(sublist)

    if remainder > 0:
        start_index = stop_index
        stop_index = start_index + remainder
        sublist = list_input[start_index:stop_index]
        sublists.append(sublist)

    return sublists

def split_list_into_sublists(lst, n):
    if not isinstance(lst, list) or not isinstance(n, int) or n <= 0:
        raise ValueError("Input must be a non-empty list and n must be a positive integer.")

    avg_sublist_length = len(lst) // n
    remainder = len(lst) % n

    sublists = []
    start = 0

    for i in range(n):
        end = start + avg_sublist_length + (1 if i < remainder else 0)
        sublists.append(lst[start:end])
        start = end

    return sublists

#%% define the positive bag
class BagPositive(BagSelective):
    def __init__(self, data, label=None, ids = None, file_list = [], weight = None, tresh = 0.1):
        super().__init__(data, label, ids = ids, label2include = 1, file_list = file_list, weight = weight)

        self.tresh = tresh
        self._list_files_negative = []
        self._list_files_positive = []
        self._ZnMix_ = False

    def separate_lists(self):

        list_data_positive, list_data_negative = [], []
        list_files_positive, list_files_negative = [], []
        list2mix = matrix_to_list(self.data)

        for i in range(0, len(self.weight)):
            if self.weight[i] >= 1 - self.tresh:
                list_data_negative.append(list2mix[i])
                if len(self.file_list) > 0:
                    list_files_negative.append(self.file_list[i])
                else:
                    list_files_negative.append([None])
            elif self.weight[i] <= self.tresh:
                list_data_positive.append(list2mix[i])
                if len(self.file_list) > 0:
                    list_files_positive.append(self.file_list[i])
                else:
                    list_files_positive.append([None])

        self._list_data_negative = list_data_negative
        self._list_data_positive = list_data_positive
        self._list_files_negative = list_files_negative
        self._list_files_positive = list_files_positive

    def plot_stats(self):
        # Call the parent class's my_function using super()
        super().plot_stats()
        if self._ZnMix_:
            #self.separate_lists()
            print(f"{len(self._list_files_negative)} instances are in the negative sub-bag")
            print(f"{len(self._list_files_positive)} instances are in the positive sub-bag")

    def remix_it(self, ratio = None, length_lists = None):

        self.separate_lists()
        if ratio is None:
            ratio = len(self._list_data_positive) / (len(self._list_data_negative) + len(self._list_data_positive) )
            ratio = round(ratio, 1)

        if ratio < 0 or ratio > 1:
            print(f"Given ratio is {ratio}")
            raise ValueError("Ratio must be a value between 0 and 1.")
        self.ratio = ratio

        if length_lists is None:
            _, counts = np.unique(self.ids, return_counts=True)
            length_lists = round(np.mean(counts))

        list_negative = list(zip(self._list_data_negative, self._list_files_negative))
        list_positive = list(zip(self._list_data_positive, self._list_files_positive))

        n_positive = int(length_lists * ratio)
        n_negative = length_lists - n_positive
        self.n_positive = n_positive
        self.n_negative = n_negative

        n = len(np.unique(self.ids))
        list_positive = split_list_into_sublists(list_positive, n)
        list_negative = split_list_into_sublists(list_negative, n)

        list_remixed, ids = [], []
        n = 0
        for i in range(0, len(list_positive)):
            sublist = list_positive[i] + list_negative[i]
            random.shuffle(sublist)
            list_remixed.append(sublist)
            ids.append(torch.ones(len(sublist)) * n)
            n += 1

        data, file_list = [], []
        for i in range(0, len(list_remixed)):
            t_bag = list_remixed[i]
            t_bag = list(zip(*t_bag))
            data.append(list(t_bag[0]))
            file_list.append(list(t_bag[1]))
        data = unnest_list(data)
        file_list = unnest_list(file_list)

        self.data = torch.stack(data)
        self.ids = torch.cat(ids, dim = 0)

        if are_all_elements_none(file_list):
            self.file_list = []
        else:
            self.file_list = file_list

        self.label = torch.ones(len(np.unique(self.ids))) * int(np.unique(self.label))
        self._ZnMix_ = True

