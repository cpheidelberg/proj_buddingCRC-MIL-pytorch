
import torch
import torchvision
import os
from tqdm import tqdm
import cv2
import platform
import numpy as np

if platform.system()=="Darwin":
    sds_path ="/Volumes/root"
else:
    sds_path = "/home/usr/root/dir"

def download_mnist_images(output_folder):
    # Create output folders per class
    for class_label in range(10):
        class_folder = os.path.join(output_folder, str(class_label))
        os.makedirs(class_folder, exist_ok=True)

    # Download and save images into respective class folders
    transform = torchvision.transforms.ToPILImage()
    mnist_dataset = torchvision.datasets.MNIST(root=output_folder, train=True, download=True)
    for i, (image, label) in tqdm(enumerate(mnist_dataset)):
        class_folder = os.path.join(output_folder, str(label))
        image_filename = f'{i}.png'
        image_path = os.path.join(class_folder, image_filename)

        image = cv2.cvtColor(np.array(image),cv2.COLOR_GRAY2RGB)
        cv2.imwrite(image_path, image)

        if (i + 1) % 1000 == 0:
            print(f'Saved {i + 1} images.')

    print('Download completed.')

# Define the output folder to save the images
output_folder = sds_path + "/DataBaseMINST" # Replace with the desired output folder path

# Call the function to download and save the images
download_mnist_images(output_folder)