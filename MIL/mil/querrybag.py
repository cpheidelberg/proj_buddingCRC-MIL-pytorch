from mil.bags import Bag
from mil.mil import MilDataset
import torch
import random
import seaborn as sns
import matplotlib.pyplot as plt
import time
import numpy as np
import h5py
class QuerryBag(Bag): #, MilDataset):
    def __init__(self, data=[], label=[], ids=None, file_list=[], bag_size = None, normalize = False, data_name = None, use_data_dump = True):
        super().__init__(data, label, ids, file_list, normalize, data_name = data_name, use_data_dump=use_data_dump)

        self.bag_size = bag_size
        self._use_refill = False

    def __getitem__(self, index):

        # get the entire bag
        t0 = time.time()
        ids = self.ids.squeeze()
        labels = self.label[index]

        if torch.is_tensor(self.data):
            bagids = self.ids[ids == self.bags[index]]
            index2load = (ids == self.bags[index])
            data = self.data.clone().detach()[index2load]

        elif isinstance(self.data, str):
            h5f = h5py.File(self.data, 'r')
            index2load = int(self.bags[index].item())
            data = h5f['data'][str(index2load)]
            data = torch.tensor(np.array(data))
            bagids = h5f['bagids'][str(index2load)]
            bagids = torch.tensor(np.array(bagids))
            h5f.close()

        t1 = time.time()
        # print(f"loading on item = {t1-t0}")

        if not self.file_list is None:
            index2load = self.ids == self.bags[index]
            index2load = index2load.tolist()
            file_list = [i for (i, v) in zip(self.file_list, index2load) if v]
            # print(file_list)
        else:
            file_list = None

        # use the oracle to random querry
        if not self.bag_size is None:

            #print(f"bag size is {data.size()}")
            if data.size()[0] >= self.bag_size:
                index = random.sample(range(0, data.size()[0]), self.bag_size)
            elif self._use_refill and data.size()[0] < self.bag_size:
                index = random.choices(range(0, data.size()[0]), k = self.bag_size)
                #print("caution, refill used")
            else:
                index = list(range(0, data.size()[0]))
                #print("caution, different data size used")

            #print(f"index is {index}")
            data = data[index]
            bagids = bagids[index]

            if not self.file_list is None:
                file_list = [file_list[i] for i in index]

        return data, bagids.unsqueeze(0), labels, file_list

    def __plot_bag_sizes__(self):

        ids = self.ids.squeeze()
        index = [int(i) for i in self.bags]
        bag_size1 = []
        for i in index:
            data = self.data[ids == i]
            bag_size1.append(data.size()[0])

        bag_size2 = []
        for i in range(0, len(index)):
            data,_,_,_ = self.__getitem__(i)
            bag_size2.append(data.size()[0])

        plt.hist([bag_size1, bag_size2])
        plt.show()



