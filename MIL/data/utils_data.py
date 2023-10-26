import os
import platform

def get_dataset_name(model_name, dataset2chose, params):

    sds_path = params['sds_path'][platform.system()]

    if "pretrainedModels" in model_name:
        model_name = "pretrained_ResNet"
    model_name = dataset2chose + "_" + model_name

    if params['data_generation']['epochs'] > 1:
        model_name += "_epochs" + str(params['data_generation']['epochs'])
    model_name += "_imageSize" + str(params['images']['im_size'])
    model_name = model_name + "_imageCrop" + str(params['data_generation']['use_image_crop'])
    save_path = os.path.join(sds_path, "sd18a006/DataBaseCRCProjekt", params['data_generation']['data_path'])

    return save_path, model_name

#%% get the case ID from the file name; ugly approach
import os
import re
from tqdm import tqdm
def get_slide_name(file_list):

    if len(file_list) ==0:
        return "NA"

    file_name = [os.path.dirname(i) for i in file_list]
    file_name = list(set(file_name))

    case_name = []
    for i, t_file_name in tqdm(enumerate(file_name), desc="get names"):
        t_file_name = os.path.basename(t_file_name)
        t_file_name = t_file_name[0:t_file_name.find("_HE")]
        t_file_name = re.findall(r'\d+', t_file_name)
        case_name.append("CaseID#" + t_file_name[0])

    if len(set(case_name)) > 1:
        print("caution mixed cases")

    case_name = str(list(set(case_name))[0])
    return case_name

#%% get case ID
def get_case_name(file_list):

    if len(file_list) ==0:
        return "NA"

    file_name = [os.path.dirname(i) for i in file_list]

    case_name = []
    for i, t_file_name in tqdm(enumerate(file_name), desc="get names"):
        t_file_name = os.path.basename(t_file_name)
        t_file_name = t_file_name[0:t_file_name.find("_HE")]
        t_file_name = re.findall(r'\d+', t_file_name)
        case_name.append("CaseID#" + t_file_name[0])

    if len(set(case_name)) > 1:
        print("caution mixed cases")

    #case_name = str(list(set(case_name))[0])
    return case_name
