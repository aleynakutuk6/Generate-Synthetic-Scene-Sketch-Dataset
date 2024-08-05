import copy
import numpy as np
import json
import os
import sys
import random
import cv2

from tqdm import tqdm
sys.path.append('/scratch/users/akutuk21/hpc_run/SketchNet-Tubitak-Project/')
from sknet.utils.visualize_utils import draw_sketch
from sknet.utils.sketch_utils import *


def extract_sknet_wh_ratio(data_info, save_path, set_type):
    
    save_path = os.path.join(save_path, set_type)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    
    sketches_path = os.path.join(save_path, "sketches")
    if not os.path.isdir(sketches_path):
        os.mkdir(sketches_path)
                
    ratios_folder = os.path.join(save_path, "wh_ratios")
    if not os.path.isdir(ratios_folder):
        os.mkdir(ratios_folder)
    
    with open(data_info, "r") as f:
        data = json.load(f)
        
    for file_name in tqdm(data):
        word_lst = data[file_name]["words"]  
        sketch = np.asarray(data[file_name]["sketch"])
        
        sketch_temp = copy.deepcopy(sketch)
        sketch_temp = apply_RDP(sketch_temp)
        sketch_temp = normalize(sketch_temp)
        min_x, min_y, max_x, max_y = get_relative_bounds(sketch_temp)
        w = max_x - min_x
        h = max_y - min_y
            
        for word in tqdm(word_lst):
            
            npz_path = os.path.join(sketches_path, word)
            if not os.path.isdir(npz_path):
                os.mkdir(npz_path)
            
            sk_name = word + "_" + file_name
            np.save(os.path.join(npz_path, sk_name + '.npy'), sketch)
            
            ratios_file = os.path.join(ratios_folder, word + ".json")
            if not os.path.isfile(ratios_file):
                ratios_data = {}
            else:
                with open(ratios_file, "r") as f:
                    ratios_data = json.load(f)
                    
            if h == 0.:
                ratios_data.update({sk_name: 0.})
            else:
                ratios_data.update({sk_name: w/h})
        
            with open(ratios_file, "w") as f:
                json.dump(ratios_data, f)  


data_info = "../SketchNet-Tubitak-Project/assets/sketchnet_mgt_data.json"
save_path = "../Datasets/sknet_ratios"

extract_sknet_wh_ratio(data_info, save_path, set_type="train")
