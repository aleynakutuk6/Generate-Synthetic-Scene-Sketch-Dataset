import json
import os
import sys
import random
import numpy as np

from tqdm import tqdm
from generate_scene import *
from utils import sketch
from Sketchformer.sketchformer_api import *


def pass_from_sketchformer(data_dir):
    
    model = get_model()
    
    for file_name in tqdm(os.listdir(data_dir)):
        
        X_test = []
        save_name = file_name.split(".")[0]
        qd_data = np.load(os.path.join(data_dir, file_name))
        sketch_temp = copy.deepcopy(qd_data)
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
            
        with open(os.path.join(data_dir, save_name + ".json"), "w") as f:
            json.dump(res_dict, f)


def get_result(data_dir, qd_classes):
    
    acc, cnt = 0, 0
    
    for file_name in tqdm(os.listdir(data_dir)):
        
        if os.path.isfile(os.path.join(data_dir, file_name)) and ".json" in file_name:
        
            with open(os.path.join(data_dir, file_name), "r") as f:
                res_data = json.load(f)
            
            save_name = file_name.split(".")[0]
            qd_class = save_name.split("_")[0]
            pred_cls = res_data["predicted"][0]["class"]
            pred_cls_name = qd_classes[pred_cls]
            if pred_cls_name == qd_class:
                acc += 1
            
            cnt += 1
            
        else:
            continue
    
    score = acc / cnt
    print("Avg score: {} \n".format(score))
    return score


f = open('Sketchformer/prep_data/quickdraw/list_quickdraw.txt', 'r')
lines = f.readlines()
qd_classes = [cls.replace("\n", "") for cls in lines]
    
data_dir = 'QD_sketches/test/strokes'
# pass_from_sketchformer(data_dir)

get_result(data_dir, qd_classes)