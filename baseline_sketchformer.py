import json
import os
import sys
import random
import numpy as np
import torch 

from tqdm import tqdm
from generate_scene import *
from utils import sketch

sys.path.append('/userfiles/akutuk21') 
from Sketchformer.sketchformer_api import *

def pass_from_sketchformer(stroke, is_absolute=False):
     
    X_test = [] 
    sketch_temp = copy.deepcopy(stroke)
    
    if is_absolute:
        min_x, min_y, max_x, max_y = get_absolute_bounds(sketch_temp)
        
        # align the drawing to the top-left corner, to have minimum values of 0.
        sketch_temp[:, 0] -= min_x
        sketch_temp[:, 1] -= min_y
                        
        sketch_temp = absolute_to_relative(sketch_temp)
        sketch_temp = normalize(sketch_temp)
        X_test.append(sketch_temp)
        
    else:
        sketch_temp = apply_RDP(sketch_temp)
        sketch_temp = normalize(sketch_temp)
        X_test.append(sketch_temp)
    
    object_embeds, predicted, class_scores = retrieve_embedding_and_classes_from_batch(model, X_test)
    object_embeds = object_embeds.tolist()
    class_scores = class_scores.numpy().tolist()
            
    res_dict = {
        "object_embeds": object_embeds,
        "class_scores": class_scores,
        "predicted": predicted
    }
            
    return res_dict


# using accuracy of matching classes
def get_acc_per_img(data, root_path, folder_name, txt_name, is_absolute=False, topN=1):

    found_cls = data["class_id"]
    data_path = data[f"{folder_name}_path"]
    
    f = open(txt_name, "a")
        
    with open(os.path.join(root_path, data_path), "r") as f2:
        vector_info = json.load(f2)
        
    class_scores = vector_info["class_scores"]
    
    # if folder_name == "strokes":
    #     sketch_stroke = np.load(os.path.join(root_path, data_path))
    # res_dict = pass_from_sketchformer(sketch_stroke, is_absolute=is_absolute)
    # class_scores = res_dict["class_scores"]
    
    # class_ids = class_ids_all[0][0:topN]
    
    class_ids_all = torch.argsort(torch.Tensor(class_scores), -1, descending=True)
    class_ids = class_ids_all[0:topN]
        
    if found_cls in class_ids:
        acc = 1
    else:
        acc = 0
    
    f.write("preds: {} gts: {} acc: {} \n".format(class_ids, found_cls, acc))
    f.close()
    
    if found_cls > 344:
        ignore = 1
    else:
        ignore = 0
    return acc, ignore
    


# using word embed distances
def get_dist_per_img(data, root_path, folder_name, txt_name, is_absolute=False, topN=1):
    
    with open("../generate_scene_dataset/json_files/QD_word_embeds_with_transformer.json", "r") as f:
        word_embeds = json.load(f)
        
    found_cls = data["class_id"]
        
    data_path = data[f"{folder_name}_path"]
    
    f = open(txt_name, "a")
    
    
    """
    if folder_name == "vectors":    
        with open(os.path.join(root_path, data_path), "r") as f2:
            vector_info = json.load(f2)
        
        sketch_stroke = np.asarray(vector_info["stroke"])
        
    elif folder_name == "strokes":
        sketch_stroke = np.load(os.path.join(root_path, data_path))
            
    res_dict = pass_from_sketchformer(sketch_stroke, is_absolute=is_absolute)
    class_scores = res_dict["class_scores"]
    """
    
    with open(os.path.join(root_path, data_path), "r") as f2:
        vector_info = json.load(f2)
        
    class_scores = vector_info["class_scores"]
    
    class_ids_all = torch.argsort(torch.Tensor(class_scores), -1, descending=True)
    class_ids = class_ids_all[0:topN]
        
    if found_cls not in class_ids:
        gt_embed = torch.Tensor(word_embeds[str(found_cls)])
        dist = 10000000
        for class_id in class_ids:
            pred_embed = torch.Tensor(word_embeds[str(class_id.data.item())])
            diff = pred_embed - gt_embed
            diff = diff.pow(2).sum().sqrt()
            dist = min(diff, dist)
            acc += dist
    else:
        dist = 0
        
    f.write("preds: {} gts: {} dist: {} \n".format(class_ids, found_cls, dist))
    f.close()
    
    if found_cls > 344:
        ignore = 1
    else:
        ignore = 0
        
    return dist, ignore
    


###################################################################################################################

synthetic_test = "../generate_scene_dataset/scene_coords-new/test"
cbsc_test = "../generate_scene_dataset/cbsc-sketches/test"
qd_test = "../generate_scene_dataset/QD_sketches/test"
our_test = "../generate_scene_dataset/custom-dataset/test"


# Set your hyperparameters

topN = 1
folder_name = "vectors" 
is_absolute = False
root_path = our_test

eval_relnet_with_embeds = False


data_name = root_path.split("/")[2]
txt_name = f"log_files/eval_log_skformer_{data_name}_top{topN}.txt"

f = open(txt_name, "w")
f.write('############################## Model Info ################################ \n')
f.write("model name: Sketchformer_Baseline \n")
f.write("test_data_root_path: {} \n".format(root_path))
f.write("folder_name: {} \n".format(folder_name))
f.write("is_absolute: {} \n".format(is_absolute))
f.write("eval_with_embeds: {} \n".format(eval_relnet_with_embeds))
f.write("topN: {} \n".format(topN))
f.write('############################## Evaluation Started ################################ \n')
f.close()

# Load Sketchformer Model

model = get_model()

# Load the data_info

with open(os.path.join(root_path, "data_info.json"), "r") as f:
    data_info = json.load(f)

# Get the accuracy

total_acc = 0
total_ctr = 0

for idx, data in enumerate(tqdm(data_info["data"])):
    if eval_relnet_with_embeds:
        acc, ignore = get_dist_per_img(data, root_path, folder_name, txt_name, is_absolute, topN)
            
    else:
        acc, ignore = get_acc_per_img(data, root_path, folder_name, txt_name, is_absolute, topN)
    
    if ignore == 0:
        total_acc += acc
        total_ctr += 1

f = open(txt_name, "a")   
f.write("Avg score: {} \n".format(total_acc / total_ctr)) 
f.close()


