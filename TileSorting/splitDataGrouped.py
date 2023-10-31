import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
import warnings

#%% set params
classification = 'Bo2NoGroupAug+'
tumorArea = 'Boarder'
normalized = True
rng = np.random.RandomState(1338)
cmap_data = plt.cm.tab20c
cmap_cv = plt.cm.coolwarm #['r', 'b', 'm']
n_splits = 1
booleanClassification = True

#%% select path for output
if normalized:
    modeldir = 'trainedModelsNormalized'
    tiledir = f'/home/dr1/sds_hd/sd18a006/DataBaseCRCProjekt/GrazKollektiv/ColourNormalizedTiles/Tumor{tumorArea}/'
    tileFormat = 'png'
    tileOrigin = '/home/dr1/sds_hd/sd18a006/DataBaseCRCProjekt/GrazKollektiv/GAN-Training/results/normalized_to_HEV_s1024_c128/test_60_tiles2Normalize/normalized'
else:
    modeldir = 'trainedModels'
    tiledir =  f'/home/dr1/sds_hd/sd18a006/DataBaseCRCProjekt/GrazKollektiv/Tumor{tumorArea}/'
    tileFormat = 'tif'
    tileOrigin = '/home/dr1/sds_hd/sd18a006/DataBaseCRCProjekt/GrazKollektiv/Original-Kollektiv'

modelpath = os.path.join('/home/dr1/PycharmProjects/GraMa/',modeldir,classification)
print('Model Path:', modelpath)
if not os.path.isdir(modelpath):
    os.mkdir(modelpath)
#check if there are tiles left in tiledir
remaining = glob.glob(os.path.join(tiledir,'*', '*', f'*.{tileFormat}'))
if not len(remaining) == 0:
    raise Warning('Tile Aim Folders are not empty, \n'
                  'use skript "getOut.py" to remove tiles')

#%% first: make a df with all the information about svs, UNum, Nodalstatus...
#%% get tiles with certain tumor areas
tiles = []
with open(f'{tumorArea}Tumor.txt', 'r') as f:
    for line in f:
        tiles.append(line.replace('\n', ''))
tiles = list(set(tiles))

#%% list of svs ID, as the svsID is not equal to UNum
svsID = []
for tile in tiles:
    svsID.append(os.path.basename(tile).split('_')[0])

#%% all the stuff before got saved as a df: just load and do work, the central has no meaning
patBudImgData = pd.read_pickle('centralTumor_svsPatData.pkl')

labels = []
for ID in svsID:
    case = patBudImgData.loc[patBudImgData['svs'] == ID]
    if case.empty:
        labels.append(np.NaN)
    else:
        labels.append(case.Nodal.to_list()[0])

ClassDataset = pd.DataFrame({'tile': tiles,
                             'ID': svsID,
                             'Nodal': labels})
ClassDataset.dropna(inplace=True)
ClassDataset.ID = ClassDataset.ID.astype(int)
if booleanClassification:
    ClassDataset.replace({'Nodal':2}, 1, inplace=True)
    if 2 in ClassDataset['Nodal'].unique():
        raise Warning('wrong labels: there is label(s) 2 instead of boolean 0 vs 1')

ClassDataset.sort_values('Nodal', inplace=True)
ClassDataset.reset_index(inplace=True,drop=True)
ClassDataset.to_pickle(modelpath + '/ClassificationDataset.pkl')


#%% split data grouped for cases
X = np.atleast_2d(ClassDataset.tile.to_numpy()).reshape(-1,1)
y = ClassDataset.Nodal.to_list()
groups = ClassDataset.ID.astype(int).to_list()
all_index = ClassDataset.index.to_numpy()


#%% split the data into train, val, test
gss = GroupShuffleSplit(n_splits= 1, train_size=.75, test_size=.15, random_state=42)
train_index, val_index = next(gss.split(X, y, groups))
test_index = np.setdiff1d(all_index, train_index) #get the idx of testSet as difference between all idx and idx in train, val
test_index = np.setdiff1d(test_index, val_index)

#%% visulization data plit over cases and classes: in progress
def plot_cv_indices(X, y, group, ax, n_splits, lw=50):
    """Create a sample plot for indices of a cross-validation object."""
    gss = GroupShuffleSplit(n_splits=1, train_size=.75, test_size=.15, random_state=42)
    # Generate the training/testing visualizations for each CV split
    train_index, val_index = next(gss.split(X=X, y=y, groups=group))
    all_index = np.arange(0, len(X), 1)
    test_index = np.setdiff1d(all_index, train_index)
    test_index = np.setdiff1d(test_index, val_index)
    indices = [train_index, val_index, test_index]
    # Fill in indices with the training/test groups
    indices = np.array([np.nan] * len(X))
    indices[train_index] = 0
    indices[val_index] = 1
    indices[test_index] = 2
    colors = ['r', 'b', 'y']
    labels = ['training set', 'validation set', 'test set']
    # Visualize the results
    # for i,indexlist in enumerate(indices):
    #     ax.scatter(
    #         indexlist,
    #         [0 + 0.5] * len(indexlist),
    #         c=colors[i], #,#cmap_cv,
    #         marker="_",
    #         lw=lw,
    #         label = labels[i]
    #         #cmap=cmap_cv,
    #         #vmin=-0.2,
    #         #vmax=1.2,
    #     )
    ax.scatter(
        range(len(indices)),
        [0 + 0.5] * len(indices),
        c=indices,
        marker="_",
        lw=lw,
        cmap=cmap_cv,
        vmin=-0.2,
        vmax=1.2,
    )

    # Plot the data classes and groups at the end
    ax.scatter(
        range(len(X)), [0 + 1.5] * len(X), c=y, marker="_", lw=lw, cmap=cmap_data
    )

    ax.scatter(
        range(len(X)), [0 + 2.5] * len(X), c=group, marker="_", lw=lw, cmap=cmap_data
    )

    # Formatting
    yticklabels = ['Data split'] + ["class", "case"]
    ax.set(
        yticks=np.arange(1 + 2) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="CV iteration",
        ylim=[n_splits + 2.2, -0.2],
        #xlim=[0, 100],
    )
    ax.set_title("{}".format(type(gss).__name__), fontsize=15)
    return ax

fig, ax = plt.subplots()
#cv = KFold(n_splits)
plot_cv_indices(X, y, groups, ax, 1)
ax.legend(
         [Patch(color=cmap_cv(.8)), Patch(color=cmap_cv(0.2)), Patch(color=cmap_cv(1.2)), Patch(color=cmap_data(0)), Patch(color=cmap_data(0.5)), Patch(color=cmap_data(100))],
         ["Validation set", "Training set", 'Test set', 'N0', 'N1', 'N2'],
        loc=(1.02, 0.8),
     )
plt.tight_layout()
plt.savefig(os.path.join(modelpath, 'Datasplit.png'))
plt.show()


#%% sort Data according to index
dataSource = tileOrigin
aim = tiledir   # '/home/dr1/sds_hd/sd18a006/DataBaseCRCProjekt/GrazKollektiv/ColourNormalizedTiles/TumorBoarder/'
oldRawDir = '/home/dr1/sds_hd/sd18a006/DataBaseCRCProjekt/GrazKollektiv'
newRawDir = '/home/dr1/sds_hd/sd18a006/DataBaseCRCProjekt/GrazKollektiv/Original-Kollektiv'
print('sorting Train')
if booleanClassification:
    tire = '2-tiered'
else:
    tire = '3-tiered'

for i,idx in enumerate(train_index):
    tile, Nodal  = ClassDataset.loc[idx, ['tile', 'Nodal']]

    if normalized:
        MoveTile = os.path.join(dataSource, os.path.basename(tile).replace('.tif', '_fake.png'))
    else:
        MoveTile = tile.replace(oldRawDir, newRawDir)
    if os.path.isfile(MoveTile):
        os.rename(MoveTile, os.path.join(aim,tire,'train', str(Nodal.astype(int)), os.path.basename(MoveTile)))
    print(f"Sorted: {i+1} / {len(train_index)}", end="\r", flush=True)
print('')

print('sorting val')
for i, idx in enumerate(val_index):
    tile, Nodal  = ClassDataset.loc[idx, ['tile', 'Nodal']]
    if normalized:
        MoveTile = os.path.join(dataSource, os.path.basename(tile).replace('.tif', '_fake.png'))
    else:
        MoveTile = tile.replace(oldRawDir, newRawDir)
    os.rename(MoveTile, os.path.join(aim, tire, 'val', str(Nodal.astype(int)), os.path.basename(MoveTile)))
    print(f"Sorted: {i+1} / {len(val_index)}", end="\r", flush=True)
print('')

print('sorting test')
for i,idx in enumerate(test_index):
    tile, Nodal  = ClassDataset.loc[idx, ['tile', 'Nodal']]
    if normalized:
        MoveTile = os.path.join(dataSource, os.path.basename(tile).replace('.tif', '_fake.png'))
    else:
        MoveTile = tile.replace(oldRawDir, newRawDir)
    os.rename(MoveTile, os.path.join(aim, tire, 'test', str(Nodal.astype(int)), os.path.basename(MoveTile)))
    print(f"Sorted: {i+1} / {len(test_index)}", end="\r", flush=True)
print('')


#%% check distribution for class error
print('check distribution for class error')
dir = os.path.join(tiledir, tire)
for phase in ['train', 'val', 'test']:
    print('phase: ', phase)
    phaseDir = os.path.join(dir, phase)
    for status in [0,1,2]:
        print('N', status)
        Files = glob.glob(os.path.join(phaseDir, str(status), f'*.{tileFormat}'))
        for file in Files:
            fileID = os.path.basename(file).split('_')[0]
            if booleanClassification:
                checkStatus = patBudImgData.loc[patBudImgData['svs'] == fileID,['Nodal']].Nodal.astype(int).replace(2,1).item()
            else:
                checkStatus = patBudImgData.loc[patBudImgData['svs'] == fileID, ['Nodal']].Nodal.astype(int).item()
            if not status == checkStatus:
                print(f'wrong in {phase, status}: {file}')

print('all done, good luck Training')
