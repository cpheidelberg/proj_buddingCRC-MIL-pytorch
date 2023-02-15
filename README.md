# CRCanalysis_pytorch

Segmentation and classification of HE-stained colorectal carcinoma (CRC) tissue.

1. Segmentation

For segmentation we use a U-Net adapted from choosehappy. In our adaption it requires a txt-file with the palette of the annotation and a folder containing the tile pairs raw HE-tile as .tif and the matching label mask as .png with equal file names. 
At first, we generated a hdf5 file as dataset. Then you can train the U-Net on the given hdf5-file. Along with the model, "train_CRCunet.py" also generates files containing statistic output such as accuracy and loss. 

2. Classification
Before classification the HE-whole slide images (WSIs) underwent preparations: 
  - "wsiFiltering.py": tiling of the WSI and exclude tiles with few tissue
  - "segmentANDsortData.py": segmentation of the tiles with the before trained U-Net and write lists with filenames for tiles containing
    a) 75% to 95% tumor, referred to as border tumor
    b) more than 95% tumor, referred to as central tumor
  - The remaining tiles were color normalized using a Gan from Runz et al.
  - "sortNormalized2Datasets.py" with an additional excel sheet containing the clinical patient data the files were sorted for the classification into training, validation, and testing
  
 "trainModel.py" generates a ResNet152 model as output along optional statistic output with Jaccard index, F1 score, accuracy. 
 
 3. Class Activation Maps (CAMs)
 Cams were computed using an adaption from xy. The script works on a list of filenames and generates an overlay output of cam and original HE-Image.

