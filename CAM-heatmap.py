#%% load the background
from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch import topk
import time
import os
import copy
import pandas as pd
from torch.nn import functional as F
import sys
#from visFunctions import imshow
from PIL import Image
import skimage.transform
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
import glob


#df = pd.read_pickle('Vorhersagewerte_B2N.pkl')
#df.sort_values('predictive Value')

files = []
with open('CAM-C2N_filelist.txt','r') as f:
    for line in f:
        values = line.rsplit(',', 3)
        files.append(values)

newPaths = []
for file in files:
    filename = os.path.basename(file[0])
    newPath = glob.glob(f'/usr/ColourNormalizedTiles/All/*/*/{filename}')
    newPaths.append(newPath[0])


#%% define some picture transformations
normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.ToTensor(),
   normalize
])
display_transform = transforms.Compose([
   transforms.Resize((224,224))])



save_path = '/usr/trainedModelsNormalized/Nodals2Classes'    #Place for test Statistics, Model is there too

#%% load and prepare the model
model = torch.load(save_path + '/Resnet-Model.pt')
model.cuda()
model.eval()

#%%set hook inside model for getting results of lasat layer before pooling
class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = ((output.cpu()).data).numpy()
    def remove(self): self.hook.remove()

#%% function to get class activation map
def getCAM(feature_conv, weight_fc, class_idx):
    _, nc, h, w = feature_conv.shape
    cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    return [cam_img]

#%% do the work
for i,pic in enumerate(newPaths):
    image = Image.open(pic)
    fallNR = os.path.basename(pic).split('_')[0]
    tensor = preprocess(image)

    prediction_var = Variable((tensor.unsqueeze(0)).cuda(), requires_grad=True)
    final_layer = model._modules.get('layer4')

    activated_features = SaveFeatures(final_layer)
    prediction = model(prediction_var)
    pred_probabilities = F.softmax(prediction).data.squeeze()
    activated_features.remove()
    topk(pred_probabilities, 1)

    weight_softmax_params = list(model._modules.get('fc').parameters())
    weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())

    weight_softmax_params

    class_idx = topk(pred_probabilities,1)[1].int()

    overlay = getCAM(activated_features.features, weight_softmax, class_idx )

    fig, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].imshow(display_transform(image))
    axs[0].set_title(f'Fall Nr: {fallNR}')   #f'Nodal status: {pic[1]}')
    axs[1].imshow(display_transform(image))
    merge = axs[1].imshow(skimage.transform.resize(overlay[0], tensor.shape[1:3]), alpha=0.5, cmap='jet')
    fig.colorbar(merge, ax = axs[1])
    axs[1].set_title(f'Prediction: {class_idx.cpu().numpy()[0]}')
    plt.savefig(f'{save_path}/CAMs/Nodal{fallNR}_{i}.png', dpi = 400)
    plt.show()