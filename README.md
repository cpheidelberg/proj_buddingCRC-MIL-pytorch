# Segmentation and classification of HE-stained colorectal carcinoma (CRC) tissue.


**Copyright (c) 2023.**

This code has been developed as part of the "Computational Pathology Heidelberg (CPH)" project.

For detailed information on usage, redistribution, and a DISCLAIMER OF ALL WARRANTIES, please refer to the "LICENSE" file in this distribution.

**BSD Simplified License**


## Description

### Segmentation

Prepare your dataset by annotating and tiling the whole slide images into 1000x1000 pixel sized tiles. Put them into one folder containing the tile pairs: 
  - raw HE-tile as .tif 
  - matching label mask as .png with equal file name 
Make sure that all labels follow the same palette and genereate the txt-file with the palette coded in rgb.

Generate the hdf5 file from the dataset. Then you can train the U-Net on the given hdf5-file. Along with the model, "train_CRCunet.py" also generates files containing statistic output such as accuracy and loss. 

### Tiling and preprocessing
Before classification the HE-whole slide images (WSIs) need preparations: 
  - "wsiFiltering.py": tiling the WSIs and exclude tiles with few tissue
  - "segmentANDsortData.py": segmentation of the tiles with the before trained U-Net and write lists with filenames for tiles containing
    a) 75% to 95% tumor, referred to as border tumor
    b) more than 95% tumor, referred to as central tumor
  - Tiles from a) and b) were color normalized using a GAN from Runz et al.
    -> normalized tiles carry the ending '_fake.png' whereas raw tiles without normalization are '.tif' files
  - Sorting:
    - "sortNormalized2Datasets.py" with an additional excel sheet containing the clinical patient data the files were sorted for the     classification into training, validation, and testing
    - "splitDataGrouped.py": Requires a pandas dataframe containing the combination of complete tile path, WSI ID and corresponding nodal status and some path adjustments. Then you can sort the tiles for training grouped after WSI such that tiles from one WSI only appear in training or validation or testing exclusively.     
  
### Classification
With "trainModel.py" you generate a ResNet152 model along optional statistic output like Jaccard index, F1 score, accuracy. 
The Dataloader is based on the folder structure with a separate folder for training, validation and testing phase, respectively. Each is subdivided into the classes (e.g. 0 and 1 for binary nodal status classification) with the single tiles in each class respective folder. The methods presented in Tiling and preprocessing - Sorting should produce this structure for you. 


    peripheralTumorDataset/
    ├── train
    |    │── 0
    |    │    ├── tileID_spacing1.tif
    |    │    ├── tileID_spacing2.tif
    |    │    ├── tileID_spacing3.tif
    |    └── 1
    |         ├── tileID2_spacing42.tif
    |         ├── ...
    ├── val
    |    │── 0
    |    │    ├── tileID3_spacing1.tif
    |    │    ├── tileID3_spacing2.tif
    |    │    ├── tileID3_spacing3.tif
    |    └── 1
    |         ├── tileID6_spacing42.tif
    |         ├── ...
    ├── test
         │── 0
         │    ├── tileID8_spacing6.tif
         │    ├── tileID8_spacing66.tif
         │    ├── tileID8_spacing666.tif
         └── 1
              ├── tileID123_spacing42.tif
              ├── ..


### MIL
Just follow the steps

## Requirements
- Linux OS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Acknowledgements
The U-Net was forked from [segmentation_epistroma_unet](https://github.com/choosehappy/PytorchDigitalPathology/tree/master/segmentation_epistroma_unet). The GAN for color normalization is supplied at [
stainTransfer_CycleGAN_pytorch ](https://github.com/m4ln/stainTransfer_CycleGAN_pytorch).

## Contribute
Contributions are very welcome! Here's how to get involved:

1. Clone or fork the repository.
2. Make your changes or improvements.
3. Create a pull request.
4. If you find any bugs or have suggestions, please log them here as well.

(Include additional details on development environment setup, coding style, testing, and issue reporting as needed.)

---
**Author:** Daniel Rusche <br>
**Contact:** [daniel.rusche@uni-heidelberg.de](mailto:daniel.rusche@uni-heidelberg.de)
