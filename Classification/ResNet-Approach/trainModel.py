#%% load the background
from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import sys
from visFunctions import imshow
import glob


#%% setup
root = '/home/usr/GrazKollektiv/TumorBoarder'   #where the tiles are stored
saveModelPth = '/home/usr/PycharmProjects/GraMa/trainedModels'  #where to store the model and statistics
classification = 'Nodals2Cases' #what to classify

#%% prepare parameters
if 'Nodal' in classification:
    classes = np.arange(0, 3, 1)
elif 'Budding' in classification:
    classes = np.arange(0,5,1)
elif 'Progress' in classification:
    classes = [0,1]
else:
    raise ValueError('Classification must contain Nodal or Budding!')

#%% get the files
folders = list(os.walk(os.path.join(root, 'train')))[0][1]
classes = np.asarray(list(map(int, folders)))


#%% checks if GPU is available, and then decide accordingly...
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print('GPU-mode is set')
else:
    print('CPU-mode is set')


#%% define the training function
# why can't that be done within another file -> not understandable
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
    loss_history = {
        "train": [],
        "val": []}
    acc_history = {
        "train": [],
        "val": []}

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        current_lr = get_lr(optimizer)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            loss_history[phase].append(epoch_loss)
            acc_history[phase].append(epoch_acc.cpu().numpy().__float__())
            if phase == 'train':
                lr_scheduler.step(epoch_acc)

            print('{}   Loss: {:.4f}    Acc: {:.4f} lr: {:.3f}'.format(
                phase, epoch_loss, epoch_acc, current_lr))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    stats = {}
    for phase in ['train', 'val']:
        stats[phase] = {}
        stats[phase]['loss'] = loss_history[phase]
        stats[phase]['acc'] = acc_history[phase]
    return model, stats, best_acc

#%% define the model loading function
def load_model(imodel):
    #Load a pretrained model and reset final fully connected layer.
    if imodel == 'ResNet18':
        model2train = models.resnet18(pretrained=True)
        model2train.modName = imodel + '_loaded '
    if imodel == 'ResNet34':
        model2train = models.resnet34(pretrained=True)
        model2train.modName = imodel + '_loaded '
    if imodel == 'ResNet50':
        model2train = models.resnet50(pretrained=True)
        model2train.modName = imodel + '_loaded '
    if imodel == 'ResNet101':
        model2train = models.resnet101(pretrained=True)
        model2train.modName = imodel + '_loaded '
    if imodel == 'ResNet152':
        model2train = models.resnet152(pretrained=True)
        model2train.modName = imodel + '_loaded '

    return (model2train)

#%% function for preparing the database
def prep_database(inputSize, data_dir):

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize([inputSize,inputSize]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize([inputSize,inputSize]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }


    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                                 shuffle=True, num_workers=4)
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    return(dataloaders, dataset_sizes, class_names)

#%% define function for adapting a pretrained model
def adapt_model(model_ft, imodel, n_classes):
    n_classes = n_classes

    if imodel.find('vgg') == 0 or imodel.find('alex') == 0:

        n_inputs = model_ft.classifier[6].in_features

        # Add on classifier
        model_ft.classifier[6] = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, n_classes), nn.LogSoftmax(dim=1))

        total_params = sum(p.numel() for p in model_ft.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(
            p.numel() for p in model_ft.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')

    elif imodel.find('densenet') == 0:
        model_ft.classifier = nn.Linear(1024, n_classes)

    elif imodel.find('squeeznet') == 0:
        model_ft.classifier[1] = nn.Conv2d(512, n_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = n_classes

    elif imodel == 'inception':

        model_ft.aux_logits = False
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, n_classes)

    else:
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, len(class_names))

    return(model_ft)

#%% define a list of models
model_list = ['ResNet152']
save_path = saveModelPth

#%% iterate over the model list
for imodel in model_list:

    #%% Data augmentation and normalization for training
    # Just normalization for validation
    if imodel == 'inception':
        inputSize = 299
    else:
        inputSize = 224
    data_dir = root
    (dataloaders, dataset_sizes, class_names) = prep_database(inputSize, data_dir)

    #%% compute weights
    phases = {'train', 'val', 'test'}
    numNodal = {key: [] for key in phases}
    for phase in phases:
        for n in classes:
            numNodal[phase].append(len(glob.glob(os.path.join(data_dir, phase, str(n), '*.tif'))))
        totalTiles = sum(numNodal['train'])
    weights = []
    for n in range(len(classes)):
        weights.append(1 -(numNodal['train'][n]/totalTiles))
    print('Weights: ', weights)
    weightsTens = torch.tensor(weights).to('cuda')

    #%% load the model
    model2train = load_model(imodel)
    print(model2train.modName)

    #%% adapt the model
    model_ft = model2train  # for debugging re-loading can be avoided
    model_ft = adapt_model(model_ft, imodel, len(classes))
    model_ft = model_ft.to('cuda')

    #%% set the training parameter
    criterion = nn.CrossEntropyLoss(weight = weightsTens)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    lr_scheduler = ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.5, patience=7, verbose=1)
    def get_lr(opt):
        for param_group in opt.param_groups:
            return param_group['lr']
    current_lr = get_lr(optimizer_ft)


    #%% train the model
    model_ft = model_ft.to(device)
    model_ft, stats, best_acc = train_model(model_ft, dataloaders, criterion, optimizer_ft, lr_scheduler,
                            num_epochs=50)

    #% save the model, stats, weights
    if not os.path.exists(os.path.join(save_path, classification)):
        os.mkdir(os.path.join(save_path, classification))
    torch.save(model_ft, os.path.join(save_path, classification, 'Resnet-Model.pt'))
    for phase in ['train', 'val']:
        for measure in ['loss', 'acc']:
            with open(os.path.join(save_path, classification, phase + '_' + measure + '_history.txt'), "w") as f:
                for s in stats[phase][measure]:
                    f.write(str(s) + "\n")
    with open(os.path.join(save_path, classification, 'dataStats.txt'), "w") as f:
            f.write('Weights: \t' + str(weights) + "\n")
            f.write('\n')
            f.write(f'best acc: {best_acc}\n')
            f.write('Dataset \t ' + str(classes) + '\n')
            for phase in phases:
                f.write(phase + '\t' + str(numNodal[phase])  + '\n')


