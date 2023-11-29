import os
import json
import copy 

from tqdm import tqdm
        

def find_classification_acc(root_dir, qd_classes):
    
    skformer_acc, mgt_acc, total_ctr = 0, 0, 0
    for set_type in os.listdir(root_dir):
        
        if set_type not in ["train", "valid", "test"]:
            continue
        
        paths_dict = {}
        
        folder_path = os.path.join(root_dir, set_type)   
        for folder_name in os.listdir(folder_path):
            
            if folder_name not in ["vectors"]:
                continue
                
            file_path = os.path.join(folder_path, folder_name)   
            for filename in tqdm(os.listdir(file_path)):
                
                class_name, img_id, rest = filename.split("_")
                inst_id = rest.split(".")[0]
                class_id = qd_classes.index(class_name)
                        
                path = os.path.join(file_path, filename) 
                
                with open(path, "r") as f:
                    data = json.load(f)
                
                idx = int(inst_id)-1
                mgt_pred_cls = data["mgt_pred"]["class"]
                skformer_pred_cls = data["predicted"]["class"]
                
                if mgt_pred_cls == class_id:
                    mgt_acc += 1
                
                if skformer_pred_cls == class_id:
                    skformer_acc += 1
                
                total_ctr += 1
            
        print("Total cnt: ", total_ctr)
        print("Sketchformer total acc: ", skformer_acc/ total_ctr)
        print("MGT total acc: ", mgt_acc/ total_ctr)
                    
                    
f = open('/userfiles/akutuk21/Sketchformer/prep_data/quickdraw/list_quickdraw.txt', 'r')
lines = f.readlines()
qd_classes = [cls.replace("\n", "") for cls in lines]


"""
# Compare MGT and Sketchformer prediction acc

cbsc_root_dir = "cbsc-sketches"
print("Dataset {} paths are processing now...".format(cbsc_root_dir))
find_classification_acc(cbsc_root_dir, qd_classes)


scene_coords_root_dir = "scene_coords-new"
print("Dataset {} paths are processing now...".format(scene_coords_root_dir))
find_classification_acc(scene_coords_root_dir, qd_classes)
"""

