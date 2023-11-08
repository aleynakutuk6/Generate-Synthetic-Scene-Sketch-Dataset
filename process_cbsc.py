import os
import torch
import numpy as np
import json
import argparse
import copy
import random

from PIL import Image
from tqdm import tqdm
from utils.sketch import *
from utils.hparams import *
from draw_scene import *
from generate_scene import *
from Sketchformer.sketchformer_api import *

#########################################################

def default_hparams():
    hps = HParams(
        obj_size=480,
        min_object_size=0.05,
        min_objects_per_image=3,
        max_objects_per_image=5,
        max_objects_per_category=3,
        min_category_per_image=3,
        save_embeds=True,
        margin = 10,
        scale_factor=255.0
    )
    return hps
    

def load_CBSC_data(root_pth, save_pth, hps, rdp_per_obj=False, rdp_per_scene=False): 
    
    # add test or validation to the root path as well
    assert not rdp_per_obj or not rdp_per_scene
    
    image_counter = 0
    
    if hps["save_embeds"]:
        model = get_model()
      
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
                
                save_dir = os.path.join(save_pth, inst)
                if not os.path.isdir(save_dir):
                    os.mkdir(save_dir)
            
                obj_names, obj_ids = read_object_order(os.path.join(inst_dir, "position_str.txt"))
                
                if hps["save_embeds"]:
                    X_test = []
                
                sketch_bboxes = []
                raster_sketches = np.zeros((len(obj_ids), hps['obj_size'], hps['obj_size'], 1))
                sketch, sketch_divisions = [], [0]
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
                    
                    sketch.extend(obj)
                    num_sk_strokes = int(np.sum(np.asarray(obj)[..., -1] == 1))
                    sketch_divisions.append(sketch_divisions[-1] + num_sk_strokes)
                    sketch_bboxes.append([xmin, ymin, xmax, ymax])
                    
                    img_path = os.path.join(save_dir, f'{str(i+1)}_{obj_names[i]}.png')
                    
                    sketch_temp = np.asarray(obj)
                    raster_sketches[i] = draw_sketch(sketch_temp, img_path, is_absolute=True, white_bg=True, max_dim=hps['obj_size'])
            
                    if hps["save_embeds"]:
                        min_x, min_y, max_x, max_y = get_absolute_bounds(sketch_temp)
                        
                        # align the drawing to the top-left corner, to have minimum values of 0.
                        sketch_temp[:, 0] -= min_x
                        sketch_temp[:, 1] -= min_y
                        
                        sketch_relative = absolute_to_relative(sketch_temp)
                        sketch_normalized = normalize(sketch_relative)
                        X_test.append(sketch_normalized)
                        
                sketch = np.asarray(sketch)
                
                if rdp_per_scene:
                    xmin, ymin, xmax, ymax = get_absolute_bounds(sketch)
                    max_len = max(xmax - xmin, ymax - ymin)
                    
                    sketch = normalize_to_scale(sketch, is_absolute=True, scale_factor=512)
                    stroke_end_inds = np.where(sketch[:, -1] > 0.5)[0]
                    sketch = apply_RDP(sketch, is_absolute=True)
                    sketch = normalize_to_scale(sketch, is_absolute=True, scale_factor=max_len)
                    stroke_end_news = np.where(sketch[:, -1] > 0.5)[0]
                    
                    assert stroke_end_news.shape[0] == stroke_end_inds.shape[0]
                    end_maps = {}
                    for bef, aft in zip(stroke_end_inds, stroke_end_news):
                        end_maps[bef] = aft
                        
                    for i in range(1, len(begins)):
                        begins[i] = end_maps[begins[i] - 1] + 1
    
                
                scene_strokes = np.asarray(sketch)
                img_path = os.path.join(save_dir, "0_scene.png")
                scene_size = max(img_h, img_w)
                raster_scene = draw_sketch(copy.deepcopy(scene_strokes), img_path, is_absolute=True, white_bg=True, keep_size=True, max_dim=scene_size)
                
                sketch_bboxes = np.asarray(sketch_bboxes)
                sketch_bboxes = np.insert(sketch_bboxes, 0, [0.0, 0.0, img_w, img_h], axis=0)
                
                if hps["save_embeds"]:
                    object_embeds, predicted, class_scores = retrieve_embedding_and_classes_from_batch(model, X_test)
                    object_embeds = object_embeds.tolist()
                    class_scores = class_scores.numpy().tolist()
                else:
                    object_embeds = None
                    class_scores = None
                    predicted = None
                
                res_dict = {"img_id": inst,
                            "img_w": img_w, "img_h": img_h,
                            "sketch_bboxes": sketch_bboxes.tolist(),
                            "raster_sketches": raster_sketches.tolist(), 
                            "qd_class_ids": obj_ids,  
                            "scene": raster_scene.tolist(),
                            "object_divisions": sketch_divisions,
                            "scene_strokes": scene_strokes.tolist(),
                            "object_embeds": object_embeds,
                            "class_scores": class_scores,
                            "predicted": predicted}
                            
                with open(os.path.join(save_dir, "data_info.json"), "w") as f:
                    json.dump(res_dict, f)
                
                image_counter += 1
    
    return image_counter


def read_object_order(position_file):
    
    with open("Sketchformer/prep_data/quickdraw/list_quickdraw.txt", "r") as f:
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
    parser.add_argument('--dataset-dir', default='../CBSC_Data/test')
    parser.add_argument('--target-dir', default='CBSC-processed')
    parser.add_argument('--hparams', type=str)

    args = parser.parse_args()
    
    test_basename = os.path.join(args.target_dir, "test")
    
    if not os.path.isdir(args.target_dir):
        os.mkdir(args.target_dir)
    
    if not os.path.isdir(test_basename):
        os.mkdir(test_basename)
        
    hps = default_hparams()
    if args.hparams is not None:
        hps = hps.parse(args.hparams)
    hps = dict(hps.values())
        
    test_n_samples = load_CBSC_data(args.dataset_dir, test_basename, hps, rdp_per_obj=True)
    print("Saved {} images for test set".format(test_n_samples))
        
        
if __name__ == '__main__':
    main()