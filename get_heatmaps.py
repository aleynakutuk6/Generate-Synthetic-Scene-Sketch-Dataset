import os
import json
import numpy as np
import copy
import math
import cv2

from tqdm import tqdm
    

def fill_heatmap(root_dir, qd_classes, n_classes=345, scene_size=480):
    
    heatmap = np.zeros((n_classes, scene_size, scene_size, 1))
    
    for set_type in os.listdir(root_dir):
        
        if set_type not in ["train", "valid", "test"]:
            continue
            
        folder_path = os.path.join(root_dir, set_type, "vectors")    
        for filename in tqdm(os.listdir(folder_path)):
    
            with open(os.path.join(folder_path, filename), "r") as f2:
                data = json.load(f2)
            
            class_name = filename.split("_")[0]
            class_id = qd_classes.index(class_name)
            
            img_w = data["img_w"]
            img_h = data["img_h"]
            coord = data["coord"]
            
            xmin, ymin = coord[:2]
            xmax, ymax = coord[2:]
            
            xmin = math.floor(xmin * scene_size / img_w)
            xmax = math.ceil(xmax * scene_size / img_w)
            
            ymin = math.floor(ymin * scene_size / img_h)
            ymax = math.ceil(ymax * scene_size / img_h)
            
            heatmap[class_id, ymin:ymax, xmin:xmax, :] += 1
            
    np.save(os.path.join(root_dir, "heatmap.npy"), heatmap)
    

def save_heatmap_imgs(root_dir, qd_classes):
    
    save_dir = os.path.join(root_dir, "heatmaps")
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
        
    heatmap = np.load(os.path.join(root_dir, "heatmap.npy"))
    
    for cls_id in range(0, heatmap.shape[0]):
        
        class_name = qd_classes[cls_id]
        h = heatmap[cls_id, ...]
        h = h * 255 / max(h.max(), 1)
        h = h.clip(0, 255).astype(np.uint8)
        
        image_path = os.path.join(save_dir, class_name + '.png')
        cv2.imwrite(image_path, h)
        
        

f = open('/userfiles/akutuk21/Sketchformer/prep_data/quickdraw/list_quickdraw.txt', 'r')
lines = f.readlines()
qd_classes = [cls.replace("\n", "") for cls in lines]


root_dir = "scene_coords-new"
print("Dataset {} heatmaps are extracting now...".format(root_dir))
fill_heatmap(root_dir, qd_classes)
print("Saving heatmap images...")
save_heatmap_imgs(root_dir, qd_classes)









