import time
import sys
import glob
import numpy as np
import pandas as pd
import os
from sklearn import model_selection

test_set_size = .1  # what percentage of the dataset should be used as a held out validation/testing set
val_set_size = .15
dataSource = '/usr/normalizedTiles'
classification = 'central2Budding'
tiles = []
saveStatistics = 'OFF'

if 'boarder' in classification:
    root = '/usr/ColourNormalizedTiles/TumorBoarder'  # where to move the files to
    with open('boarderTumor.txt', 'r') as f:
        for line in f:
            tiles.append(line.replace('\n', ''))
    tiles = list(set(tiles))

elif 'central' in classification:
    root = '/usr/ColourNormalizedTiles/TumorCentral'  # where to move the files to
    with open('centralTumor.txt', 'r') as f:
        for line in f:
            tiles.append(line.replace('\n', ''))
    tiles = list(set(tiles))
else:
    raise ValueError('Classification must contain boarder or central!')

#%% extract Cases from excel
if 'Nodal' in classification:
    Excel = pd.read_excel('/usr/PatientData.xlsx', sheet_name='PatientData')
    FallNodal = pd.DataFrame(Excel, columns= ['UNum', 'N Routine'])     #UNum resembles the case identifier, N Routine = nodal Status (int)
    dfCleared = FallNodal.dropna()
    dfCleared['UNum'] = dfCleared['UNum'].astype(int).astype(str)
    dfCleared['N Routine'] = dfCleared['N Routine'].astype(int)
    classes = np.arange(0,3,1)

elif 'Budding' in classification:
    Excel = pd.read_excel('/usr/PatientData.xlsx', sheet_name='BuddingData')
    FallNodal = pd.DataFrame(Excel) #, columns= ['U_Nummer', 'Border II (1: < 5 budding foci; 2: 5-9 budding foci; 3: 10-19 budding foci; 4: ≥20 budding foci\n)'])  #.rename(columns={"Border II (1: < 5 budding foci; 2: 5-9 budding foci; 3: 10-19 budding foci; 4: ≥20 budding foci)": "Budding"})
    FallNodal = FallNodal.iloc[:,[2, 19]]
    FallNodal = FallNodal.rename(columns = {FallNodal.columns[1] : 'Budding'})
    FallNodal = FallNodal.rename(columns =  {FallNodal.columns[0] : 'UNum'})
    dfCleared = FallNodal.dropna()
    dfCleared['UNum'] = dfCleared['UNum'].astype(int).astype(str)
    dfCleared['Budding'] = dfCleared['Budding'].astype(int)
    classes = np.arange(0,5,1)

elif 'Progress' in classification:
    Excel = pd.read_excel('/usr/PatientData.xlsx', sheet_name='PatientData')
    FallNodal = pd.DataFrame(Excel, columns=['UNum', 'Progress'])
    dfCleared = FallNodal.dropna()
    dfCleared['UNum'] = dfCleared['UNum'].astype(int).astype(str)
    dfCleared['Progress'] = dfCleared['Progress'].astype(int)
    classes = [0,1]

else:
    raise ValueError('Classification must contain Nodal or Budding!')



#%% sort tiles into classes
usableTiles =[]
notListed = []
multiListed = []

for clas in classes:
    if not os.path.exists(os.path.join(root, str(clas))):
        os.mkdir(os.path.join(root, str(clas)))

for tile in tiles:
    tileIDX = os.path.basename(tile).split('_')[0]
    Case = dfCleared[dfCleared['UNum'].str.endswith(tileIDX)]
    normalizedTile = os.path.join(dataSource,os.path.basename(tile).replace('.tif', '_fake.png'))
    if len(Case) == 1:
        NodalStat = Case.to_numpy()[0, 1]
        if not os.path.isfile(os.path.join(root, str(NodalStat), os.path.basename(normalizedTile))):
            os.rename(normalizedTile, os.path.join(root, str(NodalStat), os.path.basename(normalizedTile)))
        usableTiles.append(normalizedTile)
    elif len(Case) == 0:
        notListed.append(normalizedTile)
    else:
        multiListed.append(normalizedTile)
    print("Sorted: %d, Notlisted: %d, Multilisted: %d, Remaining: %d " % (
    len(usableTiles), len(notListed), len(multiListed),
    len(tiles) - (len(usableTiles) + len(notListed) + len(multiListed))), end="\r", flush=True)
print('')

#%% get distribution of used cases - IDs
usableIDS = []
for file in usableTiles:
    usableIDS.append(os.path.basename(file).split('_')[0])
cases = np.unique(np.asarray(usableIDS))

nonListedIDS = []
for file in notListed:
    nonListedIDS.append(os.path.basename(file).split('_')[0])
NANcases = np.unique(np.asarray(nonListedIDS))

multiListedIDS = []
for file in multiListed:
    multiListedIDS.append(os.path.basename(file).split('_')[0])
MultiCases = np.unique(np.asarray(multiListedIDS))

#%% save Distribution to txt

date = time.strftime("%Y-%m-%d")

print(f'Anzahl der verwendbaren Fälle: {len(cases)}')

if not saveStatistics=='OFF':
    with open(f'{date}_{classification}_DataSorting.txt', 'w') as f:
        f.write('Nicht gelistet:\t' + str(len(notListed)) + '\n')
        f.write(f'Anzahl der nicht gelisteten Fälle: {len(NANcases)} \n')
        f.write('Mehrfach gelistet: \t' + str(len(multiListed)) + '\n')
        f.write(f'Mehrfach aufgeführte Fälle: {len(MultiCases)} \n')
        f.write('Effektiv nutzbare tiles: \t' + str(len(usableTiles)) + '\n')
        f.write(f'Anzahl der nutzbaren Fälle: {len(cases)}')
    with open(f'{date}_{classification}_normalized_usableIDs.txt', 'w') as f:
        for ID in cases:
            f.write(f'{ID}\n')


#%% distribute each class to train, val, test
phases={}
nodals = classes
for n in nodals:
    files = glob.glob(os.path.join(root, str(n),'*png'))
    phases["train"],phases["val"] = model_selection.train_test_split(files, test_size = test_set_size + val_set_size, train_size = 1 - test_set_size - val_set_size )
    phases['val'], phases['test'] = model_selection.train_test_split(phases['val'], test_size = test_set_size/(test_set_size + val_set_size), train_size = val_set_size/(test_set_size + val_set_size))
    print('Class: ', n, ',Tiles: ', len(files))

    for phase in phases.keys():
        if not os.path.exists(os.path.join(root,phase,str(n))):
            os.mkdir(os.path.join(root,phase,str(n)))
        print('')
        print(phase)
        for counter, tile in enumerate(phases[phase]):
            print('File ', counter, '/', len(phases[phase]), end="\r", flush=True)
            os.rename(tile, os.path.join(root, phase, str(n), os.path.basename(tile)))
        #remove empty folders - they lead to wrong size of datasets
        folders = list(os.walk(os.path.join(root,phase)))[1:]
        for folder in folders:
            # folder example: ('FOLDER/3', [], ['file'])
            if not folder[2]:
                os.rmdir(folder[0])
    print('')