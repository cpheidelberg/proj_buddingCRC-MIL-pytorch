
# Segmentation and classification of HE-stained colorectal carcinoma (CRC) tissue.

## Prerequisites
- Linux OS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Segmentation

Prepare your dataset by annotating and tiling the whole slide images into 1000x1000 pixel sized tiles. Put them into one folder containing the tile pairs: 
  - raw HE-tile as .tif 
  - matching label mask as .png with equal file name 
Make sure that all labels follow the same palette and genereate the txt-file with the palette coded in rgb.

Generate the hdf5 file from the dataset. Then you can train the U-Net on the given hdf5-file. Along with the model, "train_CRCunet.py" also generates files containing statistic output such as accuracy and loss. 

## Classification
Before classification the HE-whole slide images (WSIs) need preparations: 
  - "wsiFiltering.py": tiling the WSIs and exclude tiles with few tissue
  - "segmentANDsortData.py": segmentation of the tiles with the before trained U-Net and write lists with filenames for tiles containing
    a) 75% to 95% tumor, referred to as border tumor
    b) more than 95% tumor, referred to as central tumor
  - The remaining tiles were color normalized using a GAN from Runz et al.
  - "sortNormalized2Datasets.py" with an additional excel sheet containing the clinical patient data the files were sorted for the classification into training, validation, and testing
  
 With "trainModel.py" you generate a ResNet152 model along optional statistic output like Jaccard index, F1 score, accuracy. 

## Acknowledgements
The U-Net was forked from [segmentation_epistroma_unet](https://github.com/choosehappy/PytorchDigitalPathology/tree/master/segmentation_epistroma_unet). The GAN for color normalization is supplied at [
stainTransfer_CycleGAN_pytorch ](https://github.com/m4ln/stainTransfer_CycleGAN_pytorch).
