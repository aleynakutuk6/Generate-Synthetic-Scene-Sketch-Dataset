import os
import json
import numpy as np

from generate_scene import *
from tqdm import tqdm


def save_qd_info(class_files, set_type, sample_cnt, target_dir, sketch_size=224):
    
    save_dir = os.path.join(target_dir, set_type)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
        
    images_dir = os.path.join(save_dir, 'images')
    if not os.path.isdir(images_dir):
        os.mkdir(images_dir)
        
    strokes_dir = os.path.join(save_dir, 'strokes')
    if not os.path.isdir(strokes_dir):
        os.mkdir(strokes_dir)
    
    res_dict = {"data": []}
    
    for class_id, (qd_class, qd_class_path) in enumerate(tqdm(class_files.items())):
    
        sketches = read_quickdraw_npz(qd_class_path, partition=set_type)
        n_samples = len(sketches)
        rand_ids = random.sample(np.arange(0, n_samples-1).tolist(), sample_cnt)
        
        for rand_id in rand_ids:
            save_name = qd_class + "_" + str(rand_id)
            
            qd_data = sketches[rand_id]
            qd_data = np.asarray(qd_data)
            
            stroke_path = os.path.join(strokes_dir, save_name + '.npy')
            np.save(stroke_path, qd_data)
            
            image_path = os.path.join(images_dir, save_name + '.png')
            raster_sketch = draw_sketch(qd_data, image_path, white_bg=True, max_dim=sketch_size)
            if raster_sketch is None:
                print("Class_id: {} sample_id {} has problem!! ".format(class_id, rand_id))
                continue
            
            res_dict["data"].append({
                "class_id": class_id,
                "sample_id": rand_id, 
                "class_name": qd_class,
                "strokes_path": os.path.join('strokes', save_name + '.npy'),
                "images_path": os.path.join('images', save_name + '.png')
                })
    
    with open(os.path.join(save_dir, "data_info.json"), "w") as f:
        json.dump(res_dict, f)




qd_dataset_dir = '/datasets/quickdraw/sketchrnn/npz'
target_dir = 'QD_sketches'

tr_sample_cnt = 500
val_sample_cnt = 50
test_sample_cnt = 100

if not os.path.isdir(target_dir):
    os.mkdir(target_dir)

f = open('Sketchformer/prep_data/quickdraw/list_quickdraw.txt', 'r')
lines = f.readlines()

class_files = {}
for class_name in lines:
    class_name = class_name.replace("\n", "")
    class_files[class_name] = "{}/{}.npz".format(qd_dataset_dir, class_name)
    

# save_qd_info(class_files, 'valid', val_sample_cnt, target_dir)
# save_qd_info(class_files, 'test', test_sample_cnt, target_dir)
save_qd_info(class_files, 'train', tr_sample_cnt, target_dir)

