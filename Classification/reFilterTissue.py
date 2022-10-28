#%% configurations
datapath = '/home/dr1/sds_hd/sd18a006/DataBaseCRCProjekt/GrazKollektiv/'
grayThreshold = 50
TileSize = 1000
overlap = 0
picThreshold = 0.3 * TileSize*TileSize




# das ist aus einer Funktion von ysbecca genommen...


from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator
import numpy as np
from matplotlib import pyplot as plt
#from wsi import deephistopath
from deephistopath import filter
import os
import glob
from tifffile import imsave
from PIL import Image

Files = []  #glob.glob(os.path.join(datapath,'*','*.svs'))
with open("Filelist.txt", "r") as f:
   for line in f:
       Files.append(line.rstrip('\n'))

Files.remove('/home/dr1/sds_hd/sd18a006/DataBaseCRCProjekt/GrazKollektiv/Box IV Kolon 20x/76005_TV_M_Kolon_HE.svs')
Files.remove('/home/dr1/sds_hd/sd18a006/DataBaseCRCProjekt/GrazKollektiv/Box IX Kolon 20x/20921_TV_TEVBUL1B_K_Kolon_HE.svs')
deleted = 0
for ii, file in enumerate(Files[:810]):

    # slide = open_slide(file)
    # tiles = DeepZoomGenerator(slide,
    #                         tile_size=TileSize,
    #                         overlap=overlap)
    #
    # level = len(tiles.level_tiles) -1 # per convention -1 is the highest resolution bzw. logisch, zählt ja immer von 0
    # x_tiles, y_tiles = tiles.level_tiles[level]

    #%% create Folder for tiles
    #TileFolder = f"{file.replace('.svs', '-Tiles' + '_PT*')}"
    #KeepFolder = os.path.join(TileFolder, 'Keep')
    #TrashFolder = os.path.join(TileFolder, 'trash')
    #Folders = [TileFolder]
    # for folder in Folders:
    #     if not os.path.exists(folder):
    #         os.makedirs(folder)

    tiles = glob.glob(file.replace('.svs', '*/*.tif'))

    for jj, tile in enumerate(tiles):
            print("working on %s, File %d of %d, Tile  %d of %d, deleted Files: %d " % (os.path.basename(file), ii, len(Files), jj, len(tiles), deleted),  end="\r", flush=True)

            new_tile = np.array(Image.open(tile), dtype=np.uint8)
            # gray_tile = filter.filter_rgb_to_grayscale(new_tile)    #convert tile to grayscale
            # inverseGray = filter.filter_complement(gray_tile)       #invert grayvalues for easier computation
            boolGray = filter.filter_grays(new_tile, tolerance=15,  output_type="bool")

            if np.sum(boolGray) < picThreshold:    #sums up pixels with grayvalue above threshold so to say area with tissue
                os.remove(tile)
                deleted += 1

            # else: # count % 25 == 0:    #let us have a look at disguarded tiles
            #     Tilename = f"{os.path.join(TrashFolder, os.path.basename(datapath.rstrip('.svs')))}_({x},{y}).tif"
            #     imsave(Tilename, new_tile)

                # plt.subplot(1,4,1)
                # plt.title('tile')
                # plt.imshow(new_tile)
                #
                # plt.subplot(1,4,2)
                # plt.title('tile in grayscale')
                # plt.imshow(gray_tile, cmap='gray')
                #
                # plt.subplot(1,4,3)
                # plt.title('inverse gray')
                # plt.imshow(inverseGray,cmap='gray')
                #
                # plt.subplot(1,4,4)
                # plt.title('Bool Gray')
                # plt.imshow(boolGray,cmap='gray')
                # plt.title(f"{os.path.basename(datapath.rstrip('.svs'))}_({x},{y}).tif")
                # plt.show()
                # plt.pause(.1)
                # print(count, np.sum(boolGray))



            #print('iteration #' + str(count) +' from n=' + str(y_tiles *x_tiles))


