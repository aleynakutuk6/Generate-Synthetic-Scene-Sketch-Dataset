import os
import json

from tqdm import tqdm
    

def save_paths(root_dir, qd_classes):
    
    for set_type in os.listdir(root_dir):
        
        if set_type not in ["train", "valid", "test"]:
            continue
        
        paths_dict = {}
        
        folder_path = os.path.join(root_dir, set_type)   
        for folder_name in os.listdir(folder_path):
            
            if folder_name not in ["sketches", "images"]:
                continue
                
            file_path = os.path.join(folder_path, folder_name)   
            for filename in tqdm(os.listdir(file_path)):
                
                class_name, img_id, rest = filename.split("_")
                inst_id = rest.split(".")[0]
                class_id = qd_classes.index(class_name)
                
                path = os.path.join(folder_name, filename) 
                
                if img_id not in paths_dict.keys():
                    paths_dict[img_id] = {"coord_img_paths": [], 
                                          "sketch_img_paths": [],
                                          "class_ids": {}
                                          }
                    
                if folder_name == "sketches":
                    paths_dict[img_id]["sketch_img_paths"].append(path)
                    paths_dict[img_id]["class_ids"][inst_id] = class_id
                elif folder_name == "images":
                    paths_dict[img_id]["coord_img_paths"].append(path)
                    paths_dict[img_id]["class_ids"][inst_id] = class_id
        
        
        with open(os.path.join(folder_path, "path_info.json"), "w") as f:
            json.dump(paths_dict, f)

      

f = open('/userfiles/akutuk21/Sketchformer/prep_data/quickdraw/list_quickdraw.txt', 'r')
lines = f.readlines()
qd_classes = [cls.replace("\n", "") for cls in lines]
  
cbsc_root_dir = "cbsc-sketches"
print("Dataset {} paths are extracting now...".format(cbsc_root_dir))
save_paths(cbsc_root_dir, qd_classes)


scene_coords_root_dir = "scene_coords-new"
print("Dataset {} paths are extracting now...".format(scene_coords_root_dir))
save_paths(scene_coords_root_dir, qd_classes)








