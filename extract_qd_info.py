from utils import sketch
from tqdm import tqdm
# from draw_scene import *

import copy
import numpy as np
import json
import os
    

def get_qd_classes(qd_txt_path="list_quickdraw.txt"):
    classes = []
    with open(qd_txt_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        classes.append(line.replace("\n",""))
    return classes
    

def read_quickdraw_npz(filepath: str, partition: str=None, idx=None):

    assert partition is not None or (partition is None and idx is None)
    assert partition is None or partition in ['train', 'valid', 'test']
    assert idx is None or idx == "random" or idx >= 0
    
    if not os.path.isfile(filepath):
        raise ValueError(f"No NPZ file exists in: {filepath}")
    
    sketches = np.load(filepath, allow_pickle=True, encoding="bytes")
    
    def clean_sketch(sketch: np.ndarray) -> np.ndarray:
        sk_new = sketch.astype(float)
        if sk_new[0, -1] == 1:
            sk_new = sk_new[1:, :]
        
        end_checks = sk_new[1:, -1] + sk_new[:-1, -1]
        vals = np.where(end_checks > 1)[0]
        if vals.shape[0] > 0:
            for j in np.flip(vals):
                sk_new = np.delete(sk_new, j+1, axis=0)
        
        return sk_new

    
    if partition is not None:
        sketches = sketches[partition]
        if idx is None:
            return [clean_sketch(sk) for sk in sketches]
        elif idx == "random":
            n_samples = len(sketches)
            idx = random.randint(0, n_samples-1)
            return clean_sketch(sketches[idx]), idx
        else:
            return clean_sketch(sketches[idx])
    else:
        tr = [clean_sketch(sk) for sk in sketches["train"]]
        val = [clean_sketch(sk) for sk in sketches["valid"]]
        test = [clean_sketch(sk) for sk in sketches["test"]]
        return tr, val, test


def extract_quickdraw_wh_ratio(data_path, save_path, set_type):

    obj_classes = get_qd_classes()
    
    for file_name in tqdm(os.listdir(data_path)):
        if "npz" not in file_name:
            continue

        npz_name = file_name.split(".")[0]
        data = []
        
        folder_path = os.path.join(save_path, set_type)
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)
        
        folder_path = os.path.join(folder_path, npz_name)
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)
            
        qd_data = read_quickdraw_npz(os.path.join(data_path, npz_name + ".npz"), partition=set_type, idx=None)
        
        for i in tqdm(range(0, len(qd_data))):
                
            sketch = np.asarray(qd_data[i])
            sketch_temp = copy.deepcopy(sketch)
            sketch_temp = apply_RDP(sketch_temp)
            sketch_temp = normalize(sketch_temp)
            min_x, min_y, max_x, max_y = get_relative_bounds_customized(sketch_temp)
            w = max_x - min_x
            h = max_y - min_y
            
            if h == 0.:
                data.append(0.)
            else:
                data.append(w/h)
        
        data = np.asarray(data)
        
        npy_path = os.path.join(folder_path, 'ratios.npy')
        np.save(npy_path, data)
        
        f = open(f"log_files/extract_ratios.txt", "a")
        f.write("{} done. \n".format(npz_name))
        f.close()
    
        # data = np.load(npy_path)

    f = open(f"log_files/extract_ratios.txt", "a")                             
    f.write("\n {} DONEEEEE! \n".format(set_type))
    f.write("###############################################")
    f.close()    


data_path = "/datasets/quickdraw/sketchrnn/npz/"
save_path = "qd_ratios"

set_type = "test"
extract_quickdraw_wh_ratio(data_path, save_path, set_type)
