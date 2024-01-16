import json
import os




root_dir = "custom-dataset/test/sketches"

objs_dict = {}

for file_name in os.listdir(root_dir):
    
    cls_name = file_name.split("_")[0]
    if cls_name not in objs_dict:
        objs_dict[cls_name] = 1
    else:
        objs_dict[cls_name] += 1
         
with open("custom-dataset/obj_counts.json", "w") as f:
    json.dump(objs_dict, f)
        

"""


root_dir = "scene_coords-new"
class_ids = set()
img_ids = set()

for set_type in os.listdir(root_dir):

    if set_type not in ["train", "valid", "test"]:
        continue
    
    file_path = os.path.join(root_dir, set_type, "path_info.json")   
    with open(file_path) as f:
        data = json.load(f)
    
    for img_id in data:
        classes = data[img_id]["class_ids"]
        for c in classes:
            class_ids.add(c)
        img_ids.add(img_id)
        
    
tot_img_cnt = len(list(img_ids))
tot_obj_cnt = len(list(class_ids))
print("tot_img_cnt:", tot_img_cnt)
print("tot_obj_cnt:", tot_obj_cnt)

print("class_ids: ", class_ids)

"""