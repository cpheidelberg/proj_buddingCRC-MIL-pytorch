
#%% import section
import pandas as pd

# load the dataframe
df = pd.read_pickle("data/CRC/svs-patData-BAGS.pkl")

#%% add the case ID (derived from the file names) and save the registry
from data.utils_data import get_slide_name
case_names = [get_slide_name(i) for i in df.centralTiles]
df['caseID'] = case_names
df.Budding = df.Budding.astype("Int32")
df.Nodal = df.Nodal.astype("Int32")
df.Grading = df.Grading.astype("Int32")
df.to_excel('data/Budding/dataset.xlsx', index=False)

#%% now create three new lists and a label list
import numpy as np
from tqdm import tqdm
list_border, list_central, list_tumor = [],[], []
list_nontumor = []
labels = []

for i_list in tqdm(range(0,len(df))):

    list_border.append(df.boarderTiles[i_list])
    list_central.append(df.centralTiles[i_list])
    list_tumor.append(df.boarderTiles[i_list] + df.centralTiles[i_list])
    list_nontumor.append(df.NoTumorTiles[i_list])

    if df.Nodal[i_list] == 0:
        labels.append(0)
    else:
        labels.append(1)

#%% save it
df = pd.DataFrame({'Class': labels,
                   'central': list_central, 'border': list_border,
                   "tumor": list_tumor, 'nontumor': list_nontumor})
df.to_pickle('data/Budding/Budding_bags.pkl')