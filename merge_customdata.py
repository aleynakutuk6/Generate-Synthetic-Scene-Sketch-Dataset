import json
import os


root_dir = "custom-dataset"
save_filename = "scene-dataset-merged.json"
save_dict = {}

user_data_path = os.path.join(root_dir, "user_info.json")
if os.path.exists(user_data_path):
    with open(user_data_path, 'r') as f:
        user_data = json.load(f)
else:
    user_data = {"email_to_idx": {}, 
                 "userid_to_agreement": {}, 
                 "desc_to_idx": {}, 
                 "sceneid_to_desc": {}}
    
for file_ in os.listdir(os.path.join(root_dir, "exported-data-files")):
    file_path = os.path.join(root_dir, "exported-data-files", file_)
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    scene_data = data["sceneData"]
    for key in scene_data:
        desc = scene_data[key]["scene_description"]
        scene_info = scene_data[key]["scene_info"]
        user_email = scene_data[key]["user_email"]
        agreement = scene_data[key]["agreement"]
        
        if user_email not in user_data["email_to_idx"]:
            user_id = len(user_data["email_to_idx"])
            user_data["email_to_idx"][user_email] = user_id
            
        user_id = user_data["email_to_idx"][user_email]
        if user_id not in user_data["userid_to_agreement"]:
            user_data["userid_to_agreement"][user_id] = agreement
        
        if desc not in user_data["desc_to_idx"]:
            desc_id = len(user_data["desc_to_idx"])
            user_data["desc_to_idx"][desc] = desc_id
        
        scene_id = user_data["desc_to_idx"][desc]
        if scene_id not in user_data["sceneid_to_desc"]:
            user_data["sceneid_to_desc"][scene_id] = desc
        
        desc_id = user_data["desc_to_idx"][desc]
        img_id = "sceneid-" + str(desc_id) + "-userid-" + str(user_id)
        
        save_dict[img_id] = scene_info
    
with open(os.path.join(root_dir, save_filename), "w") as f:
    json.dump(save_dict, f)


with open(user_data_path, "w") as f:
    json.dump(user_data, f)
    