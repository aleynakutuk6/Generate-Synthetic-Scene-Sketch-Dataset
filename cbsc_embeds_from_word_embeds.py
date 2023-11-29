import os
import json
import numpy as np

cbsc_ids = [132, 4, 261, 266, 11, 142, 15, 16, 146, 19, 21, 150, 23, 280, 
    26, 28, 156, 285, 31, 32, 34, 163, 164, 38, 167, 166, 169, 170, 297, 44, 
    45, 48, 305, 50, 51, 309, 54, 311, 58, 315, 61, 64, 193, 194, 323, 67, 
    197, 69, 71, 328, 326, 202, 331, 76, 77, 206, 65, 80, 83, 84, 322, 342, 
    343, 90, 94, 226, 228, 106, 110, 114, 247, 251, 124, 127]
    
word_embed_path = "../generate_scene_dataset/json_files/QD_word_embeds_with_transformer.json"
cbsc_save_path = "temp_cbsc_word_embeds_with_transformer.json"


with open(word_embed_path, "r") as f:
    all_embeds = json.load(f)
    

cbsc_embeds = {}

for idx in cbsc_ids:
    cbsc_embeds[str(idx)] = all_embeds[str(idx)]
    

with open(cbsc_save_path, "w") as f:
    json.dump(cbsc_embeds, f) 
