import os
import json
import copy 

from tqdm import tqdm
 

def save_class_scores(root_dir, qd_classes, orig_data_dir):
    
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
                    
                with open(os.path.join(orig_data_dir, set_type, img_id, "data_info.json"), "r") as f:
                    orig_data_info = json.load(f)
                
                idx = int(inst_id)-1
                data["object_embeds"] = copy.deepcopy(orig_data_info["object_embeds"][idx])
                data["class_scores"] = copy.deepcopy(orig_data_info["class_scores"][idx])
                data["predicted"] = copy.deepcopy(orig_data_info["predicted"][idx])
                
                with open(path, "w") as f:
                    json.dump(data, f)
              

def save_paths(root_dir, qd_classes, additional_classes=None):
    
    for set_type in os.listdir(root_dir):
        
        if set_type not in ["train", "valid", "test"]:
            continue
        
        paths_dict = {}
        
        folder_path = os.path.join(root_dir, set_type)   
        for folder_name in os.listdir(folder_path):
            
            if folder_name not in ["sketches"]:
                continue
                
            file_path = os.path.join(folder_path, folder_name)   
            for filename in tqdm(os.listdir(file_path)):
                
                
                class_name, img_id, rest = filename.split("_")
                inst_id = rest.split(".")[0]
                
                if class_name in qd_classes:
                    class_id = qd_classes.index(class_name)
                else:
                    class_id = additional_classes[class_name]
                
                
                if img_id not in paths_dict.keys():
                    paths_dict[img_id] = {"coord_img_paths": [], 
                                          "sketch_img_paths": [],
                                          "vector_img_paths": [],
                                          "class_ids": []
                                          }
                
                path = os.path.join(folder_name, filename) 
                paths_dict[img_id]["sketch_img_paths"].append(path)
                paths_dict[img_id]["class_ids"].append(class_id)
                
                path = os.path.join("images", filename) 
                paths_dict[img_id]["coord_img_paths"].append(path)
                
                
                filename = filename.replace(".png", ".json")
                path = os.path.join("vectors", filename) 
                paths_dict[img_id]["vector_img_paths"].append(path)
        
        
        with open(os.path.join(folder_path, "path_info.json"), "w") as f:
            json.dump(paths_dict, f)

      

f = open('/userfiles/akutuk21/Sketchformer/prep_data/quickdraw/list_quickdraw.txt', 'r')
lines = f.readlines()
qd_classes = [cls.replace("\n", "") for cls in lines]



            
            
"""
# Copy obj embeddings info to path_info json file

cbsc_root_dir = "cbsc-sketches"
orig_data_dir = "CBSC-processed"
print("Dataset {} paths are extracting now...".format(cbsc_root_dir))
save_class_scores(cbsc_root_dir, qd_classes, orig_data_dir)


scene_coords_root_dir = "scene_coords-new"
orig_data_dir = "coco-records-latest"
print("Dataset {} paths are extracting now...".format(scene_coords_root_dir))
save_class_scores(scene_coords_root_dir, qd_classes, orig_data_dir)
"""


"""
# Save paths info

cbsc_root_dir = "cbsc-sketches"
print("Dataset {} paths are extracting now...".format(cbsc_root_dir))
save_paths(cbsc_root_dir, qd_classes)


scene_coords_root_dir = "scene_coords-new"
print("Dataset {} paths are extracting now...".format(scene_coords_root_dir))
save_paths(scene_coords_root_dir, qd_classes)
"""

custom_data_root_dir = "custom-dataset"
with open(os.path.join(custom_data_root_dir, "external_classes_to_idx.json"), 'r') as f:
    data = json.load(f)

print("Dataset {} paths are extracting now...".format(custom_data_root_dir))
save_paths(custom_data_root_dir, qd_classes, data["external_classes_to_idx"])





