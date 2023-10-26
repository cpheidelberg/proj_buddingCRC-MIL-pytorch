
#% import section
from mil.bags import Bag
import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
import pandas as pd
from tqdm import tqdm

def get_metrics(true_labels, predicted_labels):

    true_labels = np.array([i.detach().cpu().numpy() for i in true_labels])
    predicted_labels= np.array([i.detach().cpu().numpy() for i in predicted_labels])

    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    return accuracy, f1, recall

#%% define the function
class Oracle(Bag):
    def __init__(self, model2use, data=[], label= [], ids = None, file_list = [], weight = None, bag_size = None, tresh = 0.2):
        super().__init__(data, label, ids, file_list, weight)

        self.model = model2use
        self.bag_size = bag_size
        self.tresh = tresh
        self.bags = torch.unique(self.ids)

    def __getitem__(self, index):

        # get the entire bag
        ids = self.ids.squeeze()
        data = self.data[ids == self.bags[index]]
        bagids = self.ids[ids == self.bags[index]]
        labels = self.label[index]

        if not self.file_list is None:
            index = ids == self.bags[index]
            index = index.tolist()
            file_list = [i for (i, v) in zip(self.file_list, index) if v]
        else:
            file_list = None

        # use the oracle to random querry
        if not self.bag_size is None:

            if data.size()[0] >= self.bag_size:
                index = random.sample(range(0, data.size()[0]), self.bag_size)
            elif self._use_refill and data.size()[0] < self.bag_size:
                index = random.choices(range(0, data.size()[0]), k = self.bag_size)
            else:
                index = list(range(0, data.size()[0]))

            data = data[index]
            bagids = bagids[index]

            if not self.file_list is None:
                file_list = [file_list[i] for i in index]

        return data, bagids.unsqueeze(0), labels, file_list

    def __analyze__(self):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        true_labels, predicted_labels = [],[]
        vip_files = []
        for i in tqdm(range(0, len(np.unique(self.ids))), desc = "bags"):

            data, bagids, labels, file_list = self.__getitem__(i)

            # do the prediction
            pred = self.model((data.to(device),
                          bagids.to(device))).squeeze()
            pred = torch.argmax(pred)
            true_labels.append(labels)
            predicted_labels.append(pred)

            # look for the decision quality
            t_df = self.model.get_decision_df(data, labels=labels, file_list=file_list)
            t_df['ids'] = i
            if i == 0:
                df = t_df
            else:
                df = pd.concat((df, t_df))

            idx = t_df.mse <= self.tresh
            vip_files.append(t_df.files[idx].to_list())

        accuracy, f1, _ = get_metrics(true_labels=true_labels, predicted_labels=predicted_labels)

        return accuracy, f1, vip_files