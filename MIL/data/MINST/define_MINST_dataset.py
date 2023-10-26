
#%% import section
import os
import glob
import pandas as pd
import platform
def find_image_files(folder_path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif']  # Add more extensions if needed
    image_files = []

    # Traverse through the folder and its subfolders
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_extension = os.path.splitext(file)[1].lower()
            if file_extension in image_extensions:
                image_files.append(os.path.join(root, file))

    return image_files

#%% get all images
if platform.system()=="Darwin":
    sds_path ="/Volumes/dir"
else:
    sds_path = "/home/usr/root/dir"

image_list = find_image_files(sds_path + "/DataBaseMINST")

#%% define the two sets
import os
def separate_files_by_folder(filenames, folder):
    folder_files = []
    remaining_files = []

    for filename in filenames:
        if os.path.basename(os.path.dirname(filename)) == folder:
            folder_files.append(filename)
        else:
            remaining_files.append(filename)

    return folder_files, remaining_files

files_class1, files_rest = separate_files_by_folder(image_list, "Letter8")

#%% now create two lists of lists
import random
n_total = 500
n_positive = 50
def combine_lists(list1, list2):
    random_list1 = random.sample(list1, n_total-n_positive)
    random_list2 = random.sample(list2, n_positive)
    new_list = random_list1 + random_list2
    return random.sample(new_list, len(new_list))

n = 250
list_files = []
label =[]
for i in range(0,n//2):
    list_files.append(combine_lists(files_rest, files_class1))
    label.append(1)
for i in range(0,n//2):
    list_files.append(random.sample(files_rest, n_total))
    label.append(0)

df = pd.DataFrame({'Nodal': label, 'MINST': list_files})

df.to_pickle('data/MINST/MINST_bags_' + str(n) + '.pkl')
print("all done ....")