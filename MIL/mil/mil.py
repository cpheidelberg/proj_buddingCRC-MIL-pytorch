import pandas as pd
from torch import nn
import torch
import numpy as np
from torch.utils.data import Dataset
import time

# %% DataSet-Definition
class MilDataset(Dataset):
    '''
    Subclass of torch.utils.data.Dataset.

    Args:
      data:
      ids:
      labels:
      normalize:
    '''

    def __init__(self, data=None, ids=None, labels=None, file_list=None, normalize=True, Bag=None):

        if not Bag is None:
            data, ids, labels, file_list = Bag.string_up()

        self.data = data
        self.labels = labels
        self.ids = ids
        self.file_list = file_list

        # Modify shape of bagids if only 1d tensor
        if (len(ids.shape) == 1):
            ids.resize_(1, len(ids))

        self.bags = torch.unique(self.ids[0])

        # Normalize
        if normalize:
            std = self.data.std(dim=0)
            mean = self.data.mean(dim=0)
            self.data = (self.data - mean) / std

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, index):
        data = self.data[self.ids[0] == self.bags[index]]
        bagids = self.ids[:, self.ids[0] == self.bags[index]]
        labels = self.labels[index]

        if not self.file_list is None:
            index = self.ids[0] == self.bags[index]
            index = index.squeeze().tolist()
            file_list = [i for (i, v) in zip(self.file_list, index) if v]
            # print(file_list)
        else:
            file_list = None
        #print(file_list)

        return data, bagids, labels

    def n_features(self):
        return self.data.size(1)

class BagModel(nn.Module):
    '''
  Model for solving MIL problems

  Args:
    prepNN: neural network created by user processing input before aggregation function (subclass of torch.nn.Module)
    afterNN: neural network created by user processing output of aggregation function and outputing final output of BagModel (subclass of torch.nn.Module)
    aggregation_func: mil.max and mil.mean supported, any aggregation function with argument 'dim' and same behaviour as torch.mean can be used

  Returns:
    Output of forward function.
  '''

    def __init__(self, prepNN, afterNN, aggregation_func):
        super().__init__()

        self.prepNN = prepNN
        self.aggregation_func = aggregation_func
        self.afterNN = afterNN

    def forward(self, input):

        if not isinstance(input, tuple):
            input = (input, torch.zeros(input.size()[0]))

        ids = input[1]
        input = input[0]

        # Modify shape of bagids if only 1d tensor
        if (len(ids.shape) == 1):
            ids.resize_(1, len(ids))

        inner_ids = ids[len(ids) - 1]

        device = input.device

        NN_out = self.prepNN(input)

        unique, inverse, counts = torch.unique(inner_ids, sorted=True, return_inverse=True, return_counts=True)
        idx = torch.cat([(inverse == x).nonzero()[0] for x in range(len(unique))]).sort()[1]
        bags = unique[idx]
        counts = counts[idx]

        output = torch.empty((len(bags), len(NN_out[0])), device=device)

        for i, bag in enumerate(bags):
            output[i] = self.aggregation_func(NN_out[inner_ids == bag], dim=0)

        output = self.afterNN(output)

        if (ids.shape[0] == 1):
            return output
        else:
            ids = ids[:len(ids) - 1]
            mask = torch.empty(0, device=device).long()
            for i in range(len(counts)):
                mask = torch.cat((mask, torch.sum(counts[:i], dtype=torch.int64).reshape(1)))
            return (output, ids[:, mask])

    def _calc_mse_(self, input, labels=None):

        if not isinstance(input, tuple):
            input = (input, torch.zeros(input.size()[0]))

        calc_mse = lambda x, y: np.mean((x - y) ** 2)
        p_value, mse_value = [], []
        for row in range(0, input[0].size()[0]):
            i_input = (input[0][row:row + 1, :], torch.unsqueeze(torch.tensor(row), dim=0))
            i_p = self.forward(i_input)
            i_p = i_p.detach().cpu()

            if labels is None:
                t_label = torch.argmax(i_p, dim=1)
                t_label = int(t_label.numpy())
            else:
                t_label = int(labels.numpy())
                # t_label = int(labels[idx].cpu().numpy())

            i_p = i_p.numpy()
            n_classes = i_p.shape[1]

            one_hot = np.zeros(n_classes)
            one_hot[t_label] = 1

            p_value.append(i_p)
            mse_value.append(calc_mse(i_p, one_hot))

        return p_value, mse_value

    def mse(self, input, labels=None, bagids=None):

        if bagids is None:
            p_value, mse_value = self._calc_mse_(input, labels)
            return p_value, mse_value
        else:
            bagids = bagids.squeeze()
            p_value, mse_value, ids = [], [], []
            for i, id in enumerate(list(np.unique(bagids))):
                p, mse = self._calc_mse_(input[bagids == id], labels=labels[i])
                p_value.append(p)
                mse_value.append(mse)
                ids.append(bagids[bagids.squeeze() == id])

            p_value = np.concatenate(p_value, axis = 0 )
            mse_value = np.concatenate(mse_value, axis=0)
            ids = torch.cat(ids, dim = 0)

        return p_value, mse_value, ids

    def get_decision_df(self, input, labels=None, file_list=None):

        p_value, mse_value = self.mse(input, labels=labels)

        if file_list is None:
            df = pd.DataFrame({'p': p_value, 'mse': mse_value})
        else:
            df = pd.DataFrame({'p': p_value, 'mse': mse_value, 'files': file_list})
        df = df.sort_values(by='mse')

        return df

    def get_most_important(self, input, labels=None, file_list=None, tresh=0.2):

        df = self.get_decision_df(input, labels=labels, file_list=file_list)

        idx = df.mse <= tresh
        vip_list = df.files[idx]

        return list(vip_list)


def collate(batch):
    '''

  '''
    #print(batch)
    t0 = time.time()
    for i, sample in enumerate(batch):
        if i == 0:
            batch_data = sample[0]
            batch_bagids = sample[1]
            batch_labels = sample[2].unsqueeze(0)
        else:
            batch_data = torch.cat((batch_data, sample[0]),dim=0)
            batch_bagids = torch.cat((batch_bagids, sample[1]), dim=1)
            #print(batch_labels.size())
            #print(sample[2].size())
            batch_labels = torch.concatenate((batch_labels, sample[2].unsqueeze(dim=0)))
        #print(batch_labels)
    #t1 =time.time()

    #out_data = torch.cat(batch_data, dim=0)
    #out_bagids = torch.cat(batch_bagids, dim=1)
    #out_labels = torch.stack(batch_labels)
    #print(f"time {t1 - t0}")

    return batch_data, batch_bagids, batch_labels

def collate_np(batch):
    '''

  '''
    batch_data = []
    batch_bagids = []
    batch_labels = []

    for sample in batch:
        batch_data.append(sample[0])
        batch_bagids.append(sample[1])
        batch_labels.append(sample[2])

    out_data = torch.cat(batch_data, dim=0)
    out_bagids = torch.cat(batch_bagids, dim=1)
    out_labels = torch.tensor(batch_labels)

    return out_data, out_bagids, out_labels
