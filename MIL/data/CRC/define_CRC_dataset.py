
#%% import section
import pandas as pd

# load the dataframe
df = pd.read_pickle("data/CRC/svs-patData-BAGS.pkl")

#%% now create two new lists
import random
import numpy as np
from tqdm import tqdm
def generate_with_probability(probability):
    """
    Generates True or False with the given probability.
    """
    return random.random() < probability

mixedList = df.NoTumorTiles.to_list()
labels = []
f = 0.1
for i_list in tqdm(range(0,len(mixedList))):

    if (generate_with_probability(0.5)) and (not df.centralTiles[i_list] is None):

        t_sub_list_target= mixedList[i_list]
        t_sub_list_source = df.centralTiles[i_list]

        if len(t_sub_list_source) == 0:
            labels.append(0)
            continue

        n = int(len(t_sub_list_target) * f)
        idx_target = np.array(random.sample(list(range(0, len(t_sub_list_target))), n))
        if n > len(t_sub_list_source):
            idx_source = np.array(random.choices(list(range(0, len(t_sub_list_source))), k = n))
        else:
            idx_source = np.array(random.sample(list(range(0, len(t_sub_list_source))), n))

        for i, j in zip(idx_target, idx_source):
            t_sub_list_target[i] = t_sub_list_source[j]

        labels.append(1)
        mixedList[i_list] = t_sub_list_target

    else:
        labels.append(0)

#%% save it
df = pd.DataFrame({'Class': labels, 'CRC': mixedList})
df.to_pickle(f'data/CRC/CRC_bags_f{f}.pkl')