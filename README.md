# CRCanalysis_pytorch

Segmentation and classification of HE-stained colorectal carcinoma (CRC) tissue.

1. Segmentation

For segmentation we use a U-Net adaptded from choosehappy. In our adaption it requires a txt-file with the palette of the annotation and a folder containing the tile pairs raw HE-tile as .tif and the matching label mask as .png with equal file names. 
At first you have to generate a hdf5 file as dataset. Then you can train the U-Net on the given hdf5-file. Along with the model, "train_CRCunet.py" also generates files containing statistic output such as accuracy and loss. 

2. Classification
Before classification the HE-whole slide images (WSIs) underwent preparations: 
  - "wsiFiltering.py" tiling of the WSI and exclude tiles with few tissue
  - segmentation of the tiles with the before trained U-Net
  - write lists with filenames for tiles containing
    a) 75% to 95% tumor, refered to as border tumor
    b) more than 95% tumor, refered to as central tumor
  - With an additional excelsheet containing the clinical patient data the files were sorted for the classification into training, validation and testing
  
 TrainResNet generates a ResNet152 model as output along statistic output with jaccard index, F1 score, accuracy. 
 
 3. Class Activation Maps (CAMs)
 Cams were computed using an adaption from xy. The script works on  a list of filenames and genereates an overlay output of cam and original HE-Image. 
