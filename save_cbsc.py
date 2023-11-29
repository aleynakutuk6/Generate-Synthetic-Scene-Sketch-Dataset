import os
import torch
import numpy as np
import json
import argparse
import copy
import random
import cv2        

from PIL import Image
from tqdm import tqdm
from utils.sketch import *
from utils.hparams import *
from draw_scene import *
from generate_scene import *

#########################################################

def default_hparams():
    hps = HParams(
        obj_size=224,
        scene_size=480,
        min_object_size=0.05,
        min_objects_per_image=3,
        max_objects_per_image=5,
        max_objects_per_category=3,
        min_category_per_image=3,
        save_embeds=False,
        margin = 10,
        scale_factor=255.0
    )
    return hps


def make_coords_img(coords, scene_size):

    raster_coords = np.zeros((len(coords), scene_size, scene_size, 1))
    
    xmax, ymax = coords[0, 2:]
    w = math.ceil(xmax)
    h = math.ceil(ymax)
           
    for i, bbox in enumerate(coords):
        img = np.full((scene_size, scene_size, 1), 0, dtype=np.uint8)
        img[0:h, 0:w, ...] = 127.
        xmin, ymin = bbox[:2]
        xmax, ymax = bbox[2:]
        xmin = math.floor(xmin)
        ymin = math.floor(ymin)
        xmax = math.ceil(xmax)
        ymax = math.ceil(ymax)
        img[ymin:ymax, xmin:xmax, :] = 255.
        raster_coords[i] = img
        
    return raster_coords


def scale_bboxes(bboxes, img_w, img_h, new_scene_size):
    
    old_scene_size = max(img_h, img_w)
    scale_factor = new_scene_size / old_scene_size
    bboxes = bboxes * scale_factor
    bboxes = bboxes.clamp(0, new_scene_size)
    
    return bboxes
    

def load_CBSC_data(root_pth, save_pth, hps, rdp_per_obj=True): 
    
    image_counter = 0
    
    sketches_dir = os.path.join(save_pth, 'sketches')
    if not os.path.isdir(sketches_dir):
        os.mkdir(sketches_dir)
        
    images_dir = os.path.join(save_pth, 'images')
    if not os.path.isdir(images_dir):
        os.mkdir(images_dir)
        
    vectors_dir = os.path.join(save_pth, 'vectors')
    if not os.path.isdir(vectors_dir):
        os.mkdir(vectors_dir)
    
    res_dict = {"data": []}
      
    for ids in tqdm(sorted(os.listdir(root_pth))):
        ids_dir = os.path.join(root_pth, ids)
        if not ids.isnumeric():
            continue
        if not os.path.isdir(ids_dir):
            continue
            
        for env in sorted(os.listdir(ids_dir)):
            env_dir = os.path.join(ids_dir, env)
            if not os.path.isdir(env_dir):
                continue
            
            for inst in sorted(os.listdir(env_dir)):
                inst_dir = os.path.join(env_dir, inst)
                if not os.path.isdir(inst_dir):
                    continue
                if not os.path.isfile(os.path.join(inst_dir, "position_str.txt")): 
                    continue
            
                obj_names, obj_ids = read_object_order(os.path.join(inst_dir, "position_str.txt"))
                
                sketch_bboxes, sketches = [], []
                
                for i in range(0, len(obj_ids)):
                    obj = read_sketch_object(os.path.join(inst_dir, f"sketch_{i}.pts"))
                    
                    scene_img = Image.open(os.path.join(inst_dir, "screenshot.jpg"))
                    scene_img = scene_img.convert("RGB")
                    img_w, img_h = scene_img.size
                    
                    if rdp_per_obj:
                        obj = np.asarray(obj).astype(float)
                        obj_temp = copy.deepcopy(obj)
                        
                        xmin, ymin, xmax, ymax = get_absolute_bounds(obj)
                        max_len = max(xmax - xmin, ymax - ymin)
                        obj = normalize_to_scale(obj, is_absolute=True, scale_factor=256)
                        obj = apply_RDP(obj, is_absolute=True)
                        obj = normalize_to_scale(obj, is_absolute=True, scale_factor=max_len)
                        obj = obj.tolist()
                        
                    sketches.append(obj)
                    sketch_bboxes.append([xmin, ymin, xmax, ymax])
                
                sketch_bboxes = np.asarray(sketch_bboxes)
                sketch_bboxes = np.insert(sketch_bboxes, 0, [0.0, 0.0, img_w, img_h], axis=0)
                
                orig_coords = copy.deepcopy(sketch_bboxes)
                coords = torch.Tensor(sketch_bboxes)
                scaled_coords = scale_bboxes(coords, img_w, img_h, hps['scene_size'])
                coords = torch.Tensor(scaled_coords)
                
                # Coords will be processed as image
                raster_coords = make_coords_img(coords, hps['scene_size'])
                
                for idx, class_id in enumerate(obj_ids):
                    qd_class = obj_names[idx]
                    save_name = qd_class + "_" + str(inst) + "_" + str(idx+1)
                    
                    # save sketch coord image
                    image_path = os.path.join(images_dir, save_name + '.png')
                    coords_img = raster_coords[idx+1].astype(np.uint8)
                    cv2.imwrite(image_path, coords_img)
                    
                    # save sketch image using stroke-3 format
                    
                    sketches_path = os.path.join(sketches_dir, save_name + '.png')
                    sketch_temp = np.asarray(sketches[idx])
                    sketches_img = draw_sketch(sketch_temp, sketches_path, is_absolute=True, white_bg=True, max_dim=hps['obj_size'])
            
                    vector_path = os.path.join(vectors_dir, save_name + '.json')
                    vector_dict = {"img_w": img_w, "img_h": img_h, "coord": orig_coords[idx+1].tolist(), "stroke": sketch_temp.tolist()}
            
                    with open(vector_path, "w") as f:
                        json.dump(vector_dict, f)
                    
                    
                    res_dict["data"].append({
                        "class_id": class_id,
                        "class_name": qd_class,
                        "vectors_path": os.path.join('vectors', save_name + '.json'),
                        "images_path": os.path.join('images', save_name + '.png'),
                        "sketches_path": os.path.join('sketches', save_name + '.png')
                    })
                
                image_counter += 1
                
    
    with open(os.path.join(save_pth, "data_info.json"), "w") as f:
        json.dump(res_dict, f)
    
    return image_counter


def read_object_order(position_file):
    
    with open("/userfiles/akutuk21/Sketchformer/prep_data/quickdraw/list_quickdraw.txt", "r") as f:
        qd_lines = f.readlines()
        
    qd_classes = {}
    for idx, cls in enumerate(qd_lines):
        qd_classes[idx] = cls.lower().strip()
        qd_classes[cls.lower().strip()] = idx
        
    with open(position_file, "r") as f:
        lines = f.readlines()
        
    obj_names, obj_ids = [], []
    for line in lines:
        obj_name = line.replace("\t", " ").strip().split(" ")[-1].lower()
            
        # only these 2 classes do not match with quick-draw
        # a temporarily match for them
        if obj_name == "phone":
            obj_name = "cell phone"
        if obj_name == "person":
            obj_name = "yoga"
        
        obj_ids.append(qd_classes[obj_name])
        obj_names.append(obj_name)
    return obj_names, obj_ids
                    

def read_sketch_object(sketch_file):
    
    sketch = []
    with open(sketch_file, "r") as f:
        lines = f.readlines()
        
    for line in lines:
        vals = line.strip().split(" ")
        for i in range(0, len(vals), 2):
            sketch.append([int(vals[i]), int(vals[i+1]), 0])
        sketch[-1][-1] = 1
    
    return sketch
        
        
def main():
    # Parsing arguments
    parser = argparse.ArgumentParser(
        description='Prepare CBSC dataset for the network')
    parser.add_argument('--dataset-dir', default='/userfiles/akutuk21/CBSC_Data')
    parser.add_argument('--target-dir', default='cbsc-sketches')
    parser.add_argument('--hparams', type=str)

    args = parser.parse_args()
    
    test_basename = os.path.join(args.target_dir, "test")
    valid_basename = os.path.join(args.target_dir, "valid")
    
    if not os.path.isdir(args.target_dir):
        os.mkdir(args.target_dir)
    
    if not os.path.isdir(test_basename):
        os.mkdir(test_basename)
        
    if not os.path.isdir(valid_basename):
        os.mkdir(valid_basename)
        
    hps = default_hparams()
    if args.hparams is not None:
        hps = hps.parse(args.hparams)
    hps = dict(hps.values())
        
    test_n_samples = load_CBSC_data(os.path.join(args.dataset_dir, "test"), test_basename, hps, rdp_per_obj=True)
    print("Saved {} images for test set".format(test_n_samples))
    
    valid_n_samples = load_CBSC_data(os.path.join(args.dataset_dir, "validation"), valid_basename, hps, rdp_per_obj=True)
    print("Saved {} images for validation set".format(valid_n_samples))
        
        
if __name__ == '__main__':
    main()