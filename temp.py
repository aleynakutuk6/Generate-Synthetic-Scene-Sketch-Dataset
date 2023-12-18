import json
import os




root_dir = "scene_coords-new"


for set_type in os.listdir(root_dir):

    if set_type not in ["train", "valid", "test"]:
        continue
    img_ids = set()
    class_ids = set()
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
    print("set type:", set_type)
    print("tot_img_cnt:", tot_img_cnt)
    print("tot_obj_cnt:", tot_obj_cnt)