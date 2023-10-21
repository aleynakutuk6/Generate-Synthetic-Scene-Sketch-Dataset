import json
import numpy as np
import os
import argparse
import copy
import random

from rdp import rdp
from tqdm import tqdm
from utils.sketch import *
from utils.hparams import *
from draw_scene import *


#########################################################
"""
def scale_bboxes(sketch_bboxes, image_size, margin):
    
    xmin, ymin = sketch_bboxes[..., :2].min(axis=0)
    xmax, ymax = sketch_bboxes[..., 2:].max(axis=0)
    sketch_bboxes[:,0] -= xmin 
    sketch_bboxes[:,1] -= ymin 
    sketch_bboxes[:,2] -= xmin
    sketch_bboxes[:,3] -= ymin 
    w, h = (xmax - xmin), (ymax - ymin)
    rate = (image_size - (2*margin)) / max(w, h)
    normalized_bboxes = (rate * sketch_bboxes) + margin
    
    return normalized_bboxes
"""

def default_hparams():
    hps = HParams(
        obj_size=224,
        min_object_size=0.05,
        min_objects_per_image=3,
        max_objects_per_image=5,
        max_objects_per_category=3,
        min_category_per_image=3,
        canvas_w=400,
        canvas_h=300
    )
    return hps
    

def load_custom_data(data_filename, target_dir, qd_meta, hps):
    
    image_counter = 0
    external_classes_to_idx = {}
    
    save_path = os.path.join(target_dir, "test")
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
        
    with open(os.path.join(target_dir, data_filename), 'r') as f:
        data = json.load(f)
    
    for scene_num, key_id in enumerate(tqdm(data["sceneData"])):
        user_email = data["sceneData"][key_id]["user_email"]
        img_id = user_email + f"_{scene_num}"
        save_dir = os.path.join(save_path, img_id)
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
            
        scene_info = data["sceneData"][key_id]["scene_info"]
        scene_description = data["sceneData"][key_id]["scene_description"]
        agreement = data["sceneData"][key_id]["agreement"]
        
        scene, qd_class_ids = [], []
        sketch_divisions, sketch_bboxes = [0], []
        raster_sketches = np.zeros((len(coco_classes), hps['obj_size'], hps['obj_size'], 1))
        last_stroke = np.asarray([0, 0])
        for sketch_num, sketch in enumerate(scene_info):
            lines = []
            drawing = sketch["drawing"]
            label = sketch["labels"]
            for stroke in drawing:
                line = []
                for x, y in zip(stroke[0], stroke[1]):
                    line.append([x, y])
                
                # apply rdp
                simplified_line = rdp(line, epsilon=2.0)
                if len(simplified_line) > 1:
                    lines.append(simplified_line) 
            
            sketch = lines_to_strokes(lines)
            sketch_temp = copy.deepcopy(sketch)
            sketch_temp = normalize(sketch_temp)
            
            min_x, min_y, max_x, max_y = get_relative_bounds_customized(sketch_temp)
            sketch_bboxes.append([min_x, min_y, max_x, max_y])
            
            sketch[0, :2] -= last_stroke
            last_stroke[0] = lines[-1][-1][0]
            last_stroke[1] = lines[-1][-1][1]
            
            scene.extend(sketch.tolist())
            num_sk_strokes = int(np.sum(sketch[..., -1] == 1))
            sketch_divisions.append(sketch_divisions[-1] + num_sk_strokes)
            
            if label in qd_meta['qd_classes_to_idx'].keys():
                sel_id = qd_meta['qd_classes_to_idx'][label]
                
            elif label in qd_meta['coco_to_sketch'].keys():
                qd_classes = qd_meta['coco_to_sketch'][label]
                rand_id = random.randint(0, len(qd_classes)-1)
                label = qd_classes[rand_id]
                sel_id = qd_meta['qd_classes_to_idx'][label]
                
            else:
                ex_cls_ids = [c for c in range(345, 2000)]
                while True:
                    sel_id = int(random.choice(ex_cls_ids))
                    if sel_id not in external_classes_to_idx.values():
                        break
                external_classes_to_idx[label] = sel_id                  
                
            qd_class_ids.append(sel_id)
            
            img_path = os.path.join(save_dir, f'{str(sketch_num+1)}_{label}.png')
            raster_sketches[sketch_num] = draw_sketch(np.asarray(qd_data), save_dir, white_bg=True, max_dim=hps['obj_size'])
        
        sketch_divisions = np.asarray(sketch_divisions)
        
        # scale each sketch object bboxes and add scene sketch bbox
        sketch_bboxes = np.asarray(sketch_bboxes)
        canvas_h = hps["canvas_h"]
        canvas_w = hps["canvas_w"]
        sketch_bboxes = np.insert(sketch_bboxes, 0, [0.0, 0.0, canvas_w, canvas_h], axis=0)
            
        img_path = os.path.join(save_dir, "0_scene.png")
        scene_size = max(, hps["canvas_w"])
        scene = draw_sketch(np.asarray(scene), img_path, is_absolute=False, white_bg=True, keep_size=True, max_dim=scene_size)
        
        # todo: class ids needs to be inserted 
        res_dict = {"img_id": img_id, 
                    "canvas_w": canvas_w,
                    "canvas_h": canvas_h, 
                    "sketch_bboxes": sketch_bboxes.tolist(),
                    "raster_sketches": raster_sketches.tolist(),
                    "qd_class_ids": qd_class_ids,
                    "scene": scene.tolist(),
                    "object_divisions": sketch_divisions.tolist(),
                    "agreement": agreement,
                    "scene_description": scene_description}
                        
        with open(os.path.join(save_dir, "data_info.json"), "w") as f:
            json.dump(res_dict, f)
        
        add_meta = {"external_classes_to_idx": external_classes_to_idx}
        with open(os.path.join(target_dir, "external_classes_to_idx.json"), 'w') as outfile:
            json.dump(add_meta, outfile)
        
        image_counter += 1
        
    return image_counter
            
def main():
    # Parsing arguments
    parser = argparse.ArgumentParser(
        description='Prepare large dataset for chunked loading')
    parser.add_argument('--data-filename', default='scene-dataset-default-rtdb-export.json')
    parser.add_argument('--target-dir', default='custom-dataset')
    parser.add_argument('--hparams', type=str)
    
    parser.add_argument('--class-relationship', type=str, default='qd_coco_files/quickdraw_to_coco_v2.json')

    args = parser.parse_args()
    
    if not os.path.isdir(args.target_dir):
        os.mkdir(args.target_dir)
    
    hps = default_hparams()
    if args.hparams is not None:
        hps = hps.parse(args.hparams)
    hps = dict(hps.values())
    
    # load the mapping QD & COCO
    with open(args.class_relationship) as clrf:
        sketch_to_coco = json.load(clrf)
        
    coco_to_sketch, sketch_to_coco_clean = {}, {}
    for class_name, mapped in sketch_to_coco.items():
        if mapped is not None:
            coco_to_sketch[mapped] = coco_to_sketch.get(mapped, []) + [class_name]
            sketch_to_coco_clean[class_name] = mapped
    coco_classes = list(set(sketch_to_coco_clean.values()))
    coco_classes_to_idx = {c: i for i, c in enumerate(coco_classes)}
    qd_classes_to_idx = {c: i for i, c in enumerate(sketch_to_coco.keys())}

    qd_meta = {"coco_classes": coco_classes, "sketch_to_coco": sketch_to_coco_clean,
                "coco_classes_to_idx": coco_classes_to_idx, "qd_classes_to_idx": qd_classes_to_idx, 
                "coco_to_sketch": coco_to_sketch}
    with open(os.path.join(args.target_dir, "qd_coco_meta.json"), 'w') as outfile:
        json.dump(qd_meta, outfile)

    test_n_samples = load_custom_data(args.data_filename, args.target_dir, qd_meta, hps)
    print("Saved {} images for custom test set".format(test_n_samples))


if __name__ == '__main__':
    main()