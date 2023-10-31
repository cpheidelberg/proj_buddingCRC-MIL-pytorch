#%% tiling of svs whole slide images (WSI) and filtering out those without enough tissue

#%% configurations
datapath = '/home/usr/data'   #folders contining svs files
TileSize = 1000
overlap = 0
picThreshold = 0.3 * TileSize*TileSize

#%% imports
from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator
import numpy as np
from matplotlib import pyplot as plt
#from wsi import deephistopath
from deephistopath import filter
import os
import glob
from tifffile import imsave

Files = []  #glob.glob(os.path.join(datapath,'*','*.svs'))
with open("Filelist.txt", "r") as f:
   for line in f:
       Files.append(line.rstrip('\n'))


for ii, file in enumerate(Files):
    slide = open_slide(file)
    tiles = DeepZoomGenerator(slide,
                            tile_size=TileSize,
                            overlap=overlap)

    level = len(tiles.level_tiles) -1 # per convention -1 is the highest resolution bzw. logisch, zählt ja immer von 0
    x_tiles, y_tiles = tiles.level_tiles[level]

    #%% create Folder for tiles
    TileFolder = f"{file.replace('.svs', '-Tiles' + '_PT' + str(picThreshold))}"
    Folders = [TileFolder]
    for folder in Folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

    x, y = 0, 0
    count, batch_count = 0, 0
    patches, coords, labels = [], [], []
    while y < y_tiles:

        while x < x_tiles:
            print("working on %s, File %d of %d, Tile  %d of %d " % (os.path.basename(file), ii, len(Files), count, tiles.tile_count), end="\r", flush=True)
            new_tilePic = tiles.get_tile(level, (x, y))
            count += 1
            if new_tilePic.height == TileSize and new_tilePic.width == TileSize:    #UNet only takes tiles with exact same size
                new_tile = np.array(new_tilePic, dtype=np.uint8)
                boolGray = filter.filter_grays(new_tile, tolerance=15,  output_type="bool")

                if np.sum(boolGray) >= picThreshold:    #sums up pixels with grayvalue above threshold so to say area with tissue
                    Tilename = f"{os.path.join(TileFolder,os.path.basename(file.rstrip('.svs')))}_({x},{y}).tif"
                    imsave(Tilename, new_tile)  #save tiles with tissue as tif


            x +=1

        x = 0
        y += 1
