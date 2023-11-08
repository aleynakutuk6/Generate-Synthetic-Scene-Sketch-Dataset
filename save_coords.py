import os
import json
import numpy as np
import torch
import copy
import math

from generate_scene import *
from tqdm import tqdm


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


def save_coord_info(data_dir, set_type, target_dir, qd_classes, sketch_size=480):
    
    save_dir = os.path.join(target_dir, set_type)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
        
    images_dir = os.path.join(save_dir, 'images')
    if not os.path.isdir(images_dir):
        os.mkdir(images_dir)
        
    vectors_dir = os.path.join(save_dir, 'vectors')
    if not os.path.isdir(vectors_dir):
        os.mkdir(vectors_dir)
    
    res_dict = {"data": []}
    
    for img_id in os.listdir(os.path.join(data_dir, set_type)):
        
        with open(os.path.join(data_dir, set_type, img_id, "data_info.json"), "r") as f:
            data_info = json.load(f)
        
        qd_class_ids = data_info["qd_class_ids"]
        coords = data_info["sketch_bboxes"]
        orig_coords = copy.deepcopy(coords)
        
        coords = torch.Tensor(coords)
        
        img_w = data_info["img_w"]
        img_h = data_info["img_h"]
        scaled_coords = scale_bboxes(coords, img_w, img_h, sketch_size)
        coords = torch.Tensor(scaled_coords)
        
        # Coords will be processed as image
        raster_coords = make_coords_img(coords, sketch_size)
        
        for idx, class_id in enumerate(qd_class_ids):
        
            qd_class = qd_classes[class_id]
            save_name = qd_class + "_" + str(img_id) + "_" + str(idx+1)
            
            vector_path = os.path.join(vectors_dir, save_name + '.json')
            vector_dict = {"img_w": img_w, "img_h": img_h, "coord": orig_coords[idx+1]}
            
            with open(vector_path, "w") as f:
                json.dump(vector_dict, f)
            
            image_path = os.path.join(images_dir, save_name + '.png')
            coords_img = raster_coords[idx+1].astype(np.uint8)
            cv2.imwrite(image_path, coords_img)
        
            res_dict["data"].append({
                "class_id": class_id,
                "class_name": qd_class,
                "vectors_path": os.path.join('vectors', save_name + '.json'),
                "images_path": os.path.join('images', save_name + '.png')
                })
                
    with open(os.path.join(save_dir, "data_info.json"), "w") as f:
        json.dump(res_dict, f)



data_dir = 'coco-records-small'
target_dir = 'scene_coords'

if not os.path.isdir(target_dir):
    os.mkdir(target_dir)
    
f = open('Sketchformer/prep_data/quickdraw/list_quickdraw.txt', 'r')
lines = f.readlines()
qd_classes = [cls.replace("\n", "") for cls in lines]

    
save_coord_info(data_dir, 'valid', target_dir, qd_classes)
save_coord_info(data_dir, 'test', target_dir, qd_classes)
save_coord_info(data_dir, 'train', target_dir, qd_classes)

