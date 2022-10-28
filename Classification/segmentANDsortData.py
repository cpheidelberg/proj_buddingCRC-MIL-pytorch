# using the pretrained UNet this script produces a segmentation mask, calculates the area marked as tumor and saves the filename to the according list

#%% imports
import os
import glob
import numpy as np
import cv2
import torch
import sklearn.feature_extraction.image
from PIL import Image

from unet import UNet
from varname import nameof


#%% -----helper function to split data into batches

def divide_batch(l, n):
    for i in range(0, l.shape[0], n):
        yield l[i:i + n, ::]

#%% ----- parse command line arguments

class struct():
    pass
args = struct() # define a struct-element
# parser = argparse.ArgumentParser(description='Make output for entire image using Unet')
args.input_pattern = '*.tif'
patch_size = 256
batch_size = 16
OUTPUT_DIR = "/home/usr/UNetOut"
# args.resize = 1
args.croppedsize = (512,512)
args.model = "/home/usr/UNet.pth"  #pretrained UNet
args.gpuid = 0
args.force = True
args.basepath = '/home/usr/data'
stride_size = patch_size

#%% ----- load network
device = torch.device(args.gpuid if torch.cuda.is_available() else 'cpu')

checkpoint = torch.load(args.model, map_location=lambda storage,
                                                        loc: storage)  # load checkpoint to CPU and then put to device https://discuss.pytorch.org/t/saving-and-loading-torch-models-on-2-machines-with-different-number-of-gpu-devices/6666
model = UNet(n_classes=checkpoint["n_classes"], in_channels=checkpoint["in_channels"],
             padding=checkpoint["padding"], depth=checkpoint["depth"], wf=checkpoint["wf"],
             up_mode=checkpoint["up_mode"], batch_norm=checkpoint["batch_norm"]).to(device)
model.load_state_dict(checkpoint["model_dict"])
model.eval()

print(f"total params: \t{sum([np.prod(p.size()) for p in model.parameters()])}")

#%% ----- get file list

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

Files = []  #glob.glob(os.path.join(datapath,'*','*.svs'))
with open("/home/dr1/PycharmProjects/wsi-tools/wsi-tools/Filelist.txt", "r") as f:
   for line in f:
       Files.append(line.rstrip('\n'))

#%% define output categories
centralTumor = []  # list of tiles containing >= 95% Tumor
boarderTumor = []  # list of tiles with 50% - 75% Tumor
betweenTumor = []  # tiles with 75%-95% Tumor
removeFile = []

#%% generate segmented output
for F in Files:
    files = glob.glob(os.path.join(F.replace('.svs', '*'), args.input_pattern))

    # %% ------ work on files
    for ii, fname in enumerate(files):

        #fname = fname.strip()  # create Output filename
        newfname_class = fname.replace('.tif', '_output.png') 

        print(f"working on file: {os.path.basename(os.path.basename(F))}, Tile {ii} of {len(files)}")

        if not args.force and os.path.exists(newfname_class):
            print("Skipping as output file exists")
            continue

        io = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)  # read InputImage
        io = cv2.resize(io, args.croppedsize)  # format InputImage like in 'make_hdf5.py'

        io_shape_orig = np.array(io.shape)

        # add half the stride as padding around the image, so that we can crop it away later
        io = np.pad(io, [(stride_size // 2, stride_size // 2), (stride_size // 2, stride_size // 2), (0, 0)],
                    mode="reflect")

        io_shape_wpad = np.array(io.shape)

        # pad to match an exact multiple of unet patch size, otherwise last row/column are lost
        npad0 = int(np.ceil(io_shape_wpad[0] / patch_size) * patch_size - io_shape_wpad[0])
        npad1 = int(np.ceil(io_shape_wpad[1] / patch_size) * patch_size - io_shape_wpad[1])

        io = np.pad(io, [(0, npad0), (0, npad1), (0, 0)], mode="constant")

        # extract patches
        arr_out = sklearn.feature_extraction.image.extract_patches(io, (patch_size, patch_size, 3), stride_size)
        # arr_out = sklearn.feature_extraction.image.extract_patches_2d(io, (patch_size, patch_size), stride_size)
        arr_out_shape = arr_out.shape
        arr_out = arr_out.reshape(-1, patch_size, patch_size, 3)

        # in case we have a large network, lets cut the list of tiles into batches
        output = np.zeros((0, checkpoint["n_classes"], patch_size, patch_size))
        for batch_arr in divide_batch(arr_out, batch_size):
            arr_out_gpu = torch.from_numpy(batch_arr.transpose(0, 3, 1, 2) / 255).type('torch.FloatTensor').to(device)

            # ---- get results
            output_batch = model(arr_out_gpu)

            # --- pull from GPU and append to rest of output
            output_batch = output_batch.detach().cpu().numpy()

            output = np.append(output, output_batch, axis=0)

        output = output.transpose((0, 2, 3, 1))

        # turn from a single list into a matrix of tiles
        output = output.reshape(arr_out_shape[0], arr_out_shape[1], patch_size, patch_size, output.shape[3])

        if stride_size == patch_size:
            output = np.concatenate(np.concatenate(output, 1), 1)
            output = output[stride_size // 2:-stride_size // 2, stride_size // 2:-stride_size // 2, :]


        else:
            # remove the padding from each tile, we only keep the center
            output = output[:, :, stride_size // 2:-stride_size // 2, stride_size // 2:-stride_size // 2, :]

            # turn all the tiles into an image
            output = np.concatenate(np.concatenate(output, 1), 1)

        # incase there was extra padding to get a multiple of patch size, remove that as well
        output = output[0:io_shape_orig[0], 0:io_shape_orig[1], :]  # remove paddind, crop back

        # --- save output
        Arr = np.uint8(np.asarray(output.argmax(axis=2)))

        tumorAmount = sum(sum(Arr == 1))
        highThresh = 0.95* Arr.size
        midThresh = 0.75*Arr.size
        lowThresh = 0.5* Arr.size

        if  tumorAmount >= highThresh:
            centralTumor.append(fname)

        elif  midThresh < tumorAmount < highThresh:
            betweenTumor.append(fname)

        elif  lowThresh <= tumorAmount <= midThresh:
            boarderTumor.append(fname)

        else:
            removeFile.append(fname)
            os.remove(fname)



    lists = [centralTumor, boarderTumor, betweenTumor, removeFile]
    for list in lists:
        with open(os.path.join(args.basepath,nameof(list)+ '.txt'), 'w') as f:
            for s in list:
                f.write(str(s) + "\n")

