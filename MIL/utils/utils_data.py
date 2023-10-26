
from torchvision import transforms, datasets, models
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
from sklearn.utils import resample
import cv2
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import random

#%% subfunctions for image preparation
def add_g(image_array, mean=0.0, var=30):
    std = var ** 0.5
    #print(image_array)
    image_add = image_array + np.random.normal(mean, std, image_array.shape)
    image_add = np.clip(image_add, 0, 255).astype(np.uint8)
    return image_add

def flip_image(image_array):
    return cv2.flip(image_array, 1)

#%% subfunctions for noise (chaos) creation
def add_noise_byshifting(noisy_children_factor, labels):
    noise_factor_target = noisy_children_factor
    noisy_children_factor = noisy_children_factor + noisy_children_factor * (1+1/len(np.unique((labels))))
    #print("adapted noise factor is " + str(noisy_children_factor))
    idx = random.sample(range(0, len(labels)), int(len(labels)*noisy_children_factor))
    labels_mod = labels.copy()
    idx_mod = idx[::-1]
    for i in range(0, len(idx)):
        labels_mod[idx[i]] = labels[idx_mod[i]]
    return labels_mod

def add_noise(noisy_children_factor, labels):
    print("fucking noise factor " + str(noisy_children_factor))
    idx = random.sample(range(0, len(labels)), int(len(labels)*noisy_children_factor))
    label_list = list(set(labels))
    labels_mod = labels.copy()
    for i in idx:
        t_labels = label_list.copy()
        t_labels.remove(labels[i])
        labels_mod[i] = random.sample(t_labels,1)[0]
    return labels_mod

#%% subfunctions for creating order (from chaos)
def random_resample(sample_in, n):
    sample_out = resample(sample_in, n_samples=n, replace=True)
    return sample_out

def get_subsets(sample, class_id):
    sub_sample= []
    for i in sample:
        if i[1]==class_id:
            sub_sample.append(i)
    return sub_sample

def rearrange(samples, class_ids, n, num_classes = None):

    #print(num_classes)
    if num_classes is None: # standard situation
        new_sample= []
        for i_class in class_ids:
            subset = get_subsets(samples, i_class)
            subset = random_resample(subset, n)
            new_sample += subset
    else: # some classes are merged
        n_classes_merged = sum([i >= (num_classes-1) for i in class_ids])
        print("n_classes merged =" + str(n_classes_merged))
        n_classes_merged = int(n/n_classes_merged)
        print("n per merged class = " + str(n_classes_merged))
        new_sample = []
        n_sub_classes_total = 0
        for i_class in class_ids:
            subset = get_subsets(samples, i_class)
            if i_class < (num_classes-1):
                subset = random_resample(subset, n)
            else:
                if n_sub_classes_total + n_classes_merged < n:
                	subset = random_resample(subset, n_classes_merged)
                	n_sub_classes_total += n_classes_merged
                else:
                	subset = random_resample(subset, n- n_classes_merged_total)
            new_sample += subset

    return new_sample

def get_class_names(samples):

    label, classnames = [], []
    for i in range(0, len(samples)):
        label.append(samples[i][1])
        classnames.append(os.path.basename(os.path.dirname(samples[i][0])))
    return classnames, label

#%% define base class
class HistoFolderBaseClass(datasets.ImageFolder):
    def __init__(self, root, noise=0, transform=None, num_classes=None, dataname=None):
        super(HistoFolderBaseClass, self).__init__(root, transform)

        self.class_names, self.class_ids = get_class_names(self.samples)
        self.dataname = dataname
        self.num_classes = num_classes

    def plot_stats(self, save_path = None):

        class_ids = list(set(self.class_ids))
        n, id = [], []
        for i in np.unique(self.class_ids):
            n.append(self.class_ids.count(i))
            id.append(str(self.classes[i]) + "(id " + str(class_ids[i]) + ")")
            #print("n=" + str(n) + " for class "+ self.classes[i])

        sns.barplot(x = id, y = n)
        plt.xticks(rotation=90)
        plt.ylabel("N per class [n]")
        plt.xlabel("Class IDs")
        plt.tight_layout()

        if save_path is None:
            plt.show()
        else:
            if os.path.isdir(save_path) ==True:
                plt.savefig(save_path + "/class-barplot.png")
                #print('test')
            else:
                plt.savefig(save_path)

    def print_stats(self):

        print("complete dataset size = " + str(len(self.samples)))
        class_ids = np.unique(self.class_ids)
        for i in class_ids:
            n = self.class_ids.count(i)
            print("n=" + str(n) + " for class " + self.classes[i] + " (class ID#" + str(i) + ")")

    def select_classes(self, n_classes):

        def subset_classes(self, class_id):
            subset = []
            for i_sample in self.samples:
                if class_id == i_sample[1]:
                    subset.append(i_sample)
            return subset

        class_ids = resample(range(0,len(self.classes)), n_samples = n_classes, replace=False)
        print(class_ids)
        subset = []
        for class_id in class_ids:
            subset.append(subset_classes(self, class_id))

        self.samples = subset
        self.classes = [self.classes[i_class] for i_class in class_ids]
        self.class_ids = [self.class_ids[i_class] for i_class in class_ids]

    def rearrange_samples(self, n_samples):
        class_ids = list(set(self.class_ids))
        self.samples = rearrange(self.samples, class_ids, n_samples, self.num_classes)
        self.class_names, self.class_ids = get_class_names(self.samples)

    def resize_samples(self, n_samples):
        self.samples = random_resample(self.samples, n_samples)
        self.class_names, self.class_ids = get_class_names(self.samples)

#%% Histology Folder Class
class HistoFolder(HistoFolderBaseClass):
    def __init__(self, root, noise = 0, redistribute = None, phase="train", basic_aug=True, transform=None, num_classes=None):
        super(HistoFolder, self).__init__(root, transform)

        self.phase = phase
        self.basic_aug = basic_aug
        self.transform = transform
        self.num_classes = num_classes
        self.aug_func = [flip_image, add_g]
        self.file_paths = root
        self.clean = True
        self.noise = noise

        if (redistribute is not None) and (redistribute != 0):
            self.rearrange_samples(redistribute)

        label = []
        for i in range(0, len(self.samples)):
            label.append(self.samples[i][1])
        self.label = label

        if noise !=0:
            self.label_noisy = add_noise(noise, label)
        else:
            self.label_noisy = label

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, label = self.samples[index]

        if self.noise != 0:
            label = self.label_noisy[index]

        #image = self.loader(path)
        image = cv2.imread(path)

        if not self.clean:
            image1 = image
            #print(image)
            image1 = self.aug_func[0](image)
            image1 = self.transform(image1)

        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                image = self.aug_func[1](image)

        if self.transform is not None:
            image = self.transform(image)

        if self.clean:
            image1 = transforms.RandomHorizontalFlip(p=1)(image)

        # adapt the label
        if (self.num_classes is not None) and \
                (len(self.class_ids) > self.num_classes):

            if label >= self.num_classes-1:
                label = self.num_classes - 1

        return image, label, index, image1

#%% Special Budding Class
class DataSetNodal(HistoFolderBaseClass):
    def __init__(self, root, noise = 0, redistribute = None, transform=None, num_classes=None):
        super(DataSetNodal, self).__init__(root, transform)

        self.transform = transform
        self.num_classes = num_classes
        self.noise = noise
        if (redistribute is not None) and (redistribute != 0):
            self.rearrange_samples(redistribute)

        label = []
        for i in range(0, len(self.samples)):
            label.append(self.samples[i][1])
        self.label = label

        if noise !=0:
            self.label_noisy = add_noise(noise, label)
        else:
            self.label_noisy = label

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.noise != 0:
            target = self.label_noisy[index]

        if self.transform is not None:
            sample = self.transform(sample)

        # adapt the label
        if (self.num_classes is not None) and \
                (len(self.class_ids) > self.num_classes):

            if target < self.num_classes:
                target = target
            else:
                target = self.num_classes - 1

        return sample, target

#%% split data
from sklearn.model_selection import train_test_split
import torch
def create_index(list_b, list_a):
    list_b = list_b.squeeze().numpy()
    idx = list_b == list_a[0]
    for i in range(1, len(list_a)):
        idx += list_b == list_a[i]
    return idx
def split_data(instances, ids, labels, file_list, split_factor = 0.2):

    train_indices, test_indices = train_test_split(np.arange(len(labels)), test_size=split_factor)

    labels_train = labels[train_indices]
    labels_val = labels[test_indices]

    idx_train = create_index(ids, train_indices)
    idx_val = create_index(ids, test_indices)

    def split_tensors(tensor):
        tensor_train = tensor[idx_train]
        tensor_val = tensor[idx_val]
        return tensor_train, tensor_val

    instances_train, instances_val = split_tensors(instances)
    ids_train, ids_val = split_tensors(ids)

    def split_ids(ids, indices):
        ids_new = []
        for i in range(0, len(indices)):
            idx = ids == indices[i]
            t_ids = torch.ones(len(ids[idx])) * i
            ids_new.append(t_ids.long())
        ids_new = torch.cat(ids_new)

        return ids_new

    #ids_train = split_ids(ids, train_indices)
    #ids_val = split_ids(ids, test_indices)

    def split_list(list_in):
        list_train = [i for indx,i in enumerate(list_in) if idx_train[indx]]
        list_val = [i for indx, i in enumerate(list_in) if idx_val[indx]]
        return list_train, list_val

    file_list_train, file_list_val = split_list(file_list)

    return [instances_train, instances_val], [ids_train, ids_val], \
           [labels_train, labels_val], [file_list_train, file_list_val]


#%% create composite
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

# Assuming you have a batch of images 'batch'
# batch_size, channels, height, width = batch.size()
def create_map(batch, save_path):
    # Create a grid of images
    batch_size = int(np.sqrt(batch.size()[0]))
    grid_image = torchvision.utils.make_grid(batch, nrow=batch_size, ncol=batch_size)

    # Convert to NumPy array and rearrange dimensions
    composite_image = grid_image.permute(1, 2, 0).cpu().numpy()

    # Display the composite image
    plt.imshow(composite_image)
    plt.axis('off')
    plt.savefig(save_path + ".png")
    plt.show()
    plt.pause(1)
    plt.close()

#%% remove sds_path
def remove_sds_path(file_list):

    for i, i_file in enumerate(file_list):
        idx = i_file.find("DataBaseCRCProjekt")
        file_list[i] = i_file[idx:]

    return file_list