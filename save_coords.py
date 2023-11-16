import os
import json
import numpy as np
import torch
import copy
import math
import cv2

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


def save_coord_info(data_dir, set_type, target_dir, qd_classes, scene_size=480, sketch_size=224):
    
    save_dir = os.path.join(target_dir, set_type)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    
    sketches_dir = os.path.join(save_dir, 'sketches')
    if not os.path.isdir(sketches_dir):
        os.mkdir(sketches_dir)
        
    images_dir = os.path.join(save_dir, 'images')
    if not os.path.isdir(images_dir):
        os.mkdir(images_dir)
        
    vectors_dir = os.path.join(save_dir, 'vectors')
    if not os.path.isdir(vectors_dir):
        os.mkdir(vectors_dir)
    
    res_dict = {"data": []}
    
    for img_id in tqdm(os.listdir(os.path.join(data_dir, set_type))):
        
        if not os.path.exists(os.path.join(data_dir, set_type, img_id, "data_info.json")):
            print("folder {} does not have data_info.json file. \n".format(img_id))
            continue
            
        with open(os.path.join(data_dir, set_type, img_id, "data_info.json"), "r") as f:
            data_info = json.load(f)
        
        qd_class_ids = data_info["qd_class_ids"]
        coords = data_info["sketch_bboxes"]
        old_raster_sketches = data_info["raster_sketches"]
        
        raster_sketches = [] 
        for i, sketch_img in enumerate(old_raster_sketches):
            new_sketch_img = cv2.resize(np.asarray(sketch_img).astype('float32'), (sketch_size, sketch_size), interpolation = cv2.INTER_AREA)
            raster_sketches.append(new_sketch_img)
        
        raster_sketches = np.asarray(raster_sketches)
        
        orig_coords = copy.deepcopy(coords)
        coords = torch.Tensor(coords)
        
        img_w = data_info["img_w"]
        img_h = data_info["img_h"]
        scaled_coords = scale_bboxes(coords, img_w, img_h, scene_size)
        coords = torch.Tensor(scaled_coords)
        
        # Coords will be processed as image
        raster_coords = make_coords_img(coords, scene_size)
        
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
            
            sketches_path = os.path.join(sketches_dir, save_name + '.png')
            sketches_img = raster_sketches[idx].astype(np.uint8)
            cv2.imwrite(sketches_path, sketches_img)
        
            res_dict["data"].append({
                "class_id": class_id,
                "class_name": qd_class,
                "vectors_path": os.path.join('vectors', save_name + '.json'),
                "images_path": os.path.join('images', save_name + '.png'),
                "sketches_path": os.path.join('sketches', save_name + '.png')
                })
                   
    with open(os.path.join(save_dir, "data_info.json"), "w") as f:
        json.dump(res_dict, f)



data_dir = 'coco-records-latest'
target_dir = 'scene_coords'

if not os.path.isdir(target_dir):
    os.mkdir(target_dir)
    
f = open('Sketchformer/prep_data/quickdraw/list_quickdraw.txt', 'r')
lines = f.readlines()
qd_classes = [cls.replace("\n", "") for cls in lines]

    
# save_coord_info(data_dir, 'valid', target_dir, qd_classes)
# save_coord_info(data_dir, 'test', target_dir, qd_classes)
save_coord_info(data_dir, 'train', target_dir, qd_classes)

