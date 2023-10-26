
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob

data_set = "/home/cw9/sds_hd/sd21c015/Dataset_CRCvsICCA"
data_set = glob.glob(data_set + "/train/**/*jpg")

image = cv2.imread(data_set[0])
plt.imshow(image)
plt.show()

#%%
from histocartography.preprocessing import NucleiExtractor, DeepFeatureExtractor, KNNGraphBuilder
nuclei_detector = NucleiExtractor()
feature_extractor = DeepFeatureExtractor(architecture='resnet34', patch_size=72)
knn_graph_builder = KNNGraphBuilder(k=5, thresh=50, add_loc_feats=True)

nuclei_map, _ = nuclei_detector.process(image)
features = feature_extractor.process(image)
cell_graph = knn_graph_builder.process(features)