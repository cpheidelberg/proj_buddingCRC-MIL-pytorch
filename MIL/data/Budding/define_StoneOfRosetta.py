

#%% import section
import pandas as pd
df_dataset = pd.read_pickle("data/CRC/svs-patData-BAGS.pkl")    #pandas DF with columns among others...
# [centralTiles, boarderTiles, NoTumorTiles] each with lists of paths to corresponding WSI-tiles, one WSI each row

#%% get a list of all images (without sds-path)
def get_file_names(file_list):
    flat_list = []
    for i in file_list:
        flat_list+= i
    return flat_list

from utils.utils_data import remove_sds_path
list_central_tiles = get_file_names(df_dataset.centralTiles)
list_central_tiles = remove_sds_path(list_central_tiles)
list_border_tiles = get_file_names(df_dataset.boarderTiles)
list_border_tiles = remove_sds_path(list_border_tiles)

#%% adapt folder
def adapt_name(file_list):
    for i, i_file in enumerate(file_list):
        file_list[i] = i_file.replace('GrazKollektiv', 'GrazKollektiv/Original-Kollektiv')
    return file_list

list_border_tiles = adapt_name(list_border_tiles)
list_central_tiles = adapt_name(list_central_tiles)

#%% mount the stone
from data.utils_data import get_case_name
caseID = get_case_name(list_central_tiles)
df_central = pd.DataFrame({'caseID': caseID,
                           'image': list_central_tiles,
                           'location': ['central']*len(caseID)})
caseID = get_case_name(list_border_tiles)
df_border = pd.DataFrame({'caseID': caseID,
                           'image': list_border_tiles,
                           'location': ['border']*len(caseID)})

df_Rosetta = pd.concat((df_border, df_central))
df_Rosetta.to_excel("data/Budding/StoneOfRosetta.xlsx")