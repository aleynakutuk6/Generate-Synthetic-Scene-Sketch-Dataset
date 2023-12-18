import json
import numpy as np
import os
import argparse
import copy
import random
import cv2
import sys
import torch

from rdp import rdp
from tqdm import tqdm
from utils.sketch import *
from utils.hparams import *
from draw_scene import *
from generate_scene import *
sys.path.append('/userfiles/akutuk21') 
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
        img_w=1100,
        img_h=770,
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



def simplify_drawing(drawing):
    
    lines = []
    for stroke in drawing:
        line = []
        for x, y in zip(stroke[0], stroke[1]):
            line.append([x, y])
                
        # apply rdp
        simplified_line = rdp(line, epsilon=2.0)
        if len(simplified_line) > 1:
            lines.append(simplified_line)
            
    return lines



def get_cls_info(label, metadata, external_classes_to_idx):
    
    qd_classes = [c.lower() for c in metadata['qd_classes_to_idx'].keys()]
    sknet_classes = [c.lower() for c in metadata['sknet_classes_to_idx'].keys()]
    tot_obj_cnt = len(qd_classes) + len(sknet_classes)
    
    if label in qd_classes:
        sel_id = metadata['qd_classes_to_idx'][label]
        
    elif label in sknet_classes:
        sel_id = metadata['sknet_classes_to_idx'][label]
          
    elif label in metadata['coco_to_sketch'].keys():
        qd_classes = metadata['coco_to_sketch'][label]
        rand_id = random.randint(0, len(qd_classes)-1)
        label = qd_classes[rand_id]
        sel_id = metadata['qd_classes_to_idx'][label]
            
    elif label in external_classes_to_idx:
        sel_id = external_classes_to_idx[label]
                
    else:
        sel_id = tot_obj_cnt + len(external_classes_to_idx)
        external_classes_to_idx[label] = sel_id 
    
    return label, sel_id
    

def load_custom_data(data_filename, target_dir, metadata, hps):
    
    external_classes_to_idx = {}
    image_counter = 0
    res_dict = {"data": []}
    paths_dict = {}
    
    save_pth = os.path.join(target_dir, "test")
    if not os.path.isdir(save_pth):
        os.mkdir(save_pth)
        
    sketches_dir = os.path.join(save_pth, 'sketches')
    if not os.path.isdir(sketches_dir):
        os.mkdir(sketches_dir)
        
    images_dir = os.path.join(save_pth, 'images')
    if not os.path.isdir(images_dir):
        os.mkdir(images_dir)
        
    vectors_dir = os.path.join(save_pth, 'vectors')
    if not os.path.isdir(vectors_dir):
        os.mkdir(vectors_dir)
        
    scenes_dir = os.path.join(save_pth, 'scenes')
    if not os.path.isdir(scenes_dir):
        os.mkdir(scenes_dir)
        
    with open(os.path.join(target_dir, data_filename), 'r') as f:
        data = json.load(f)
    
    if hps["save_embeds"]:
        model = get_model()
    
    for img_id, scene_info in tqdm(data.items()):
        
        if hps["save_embeds"]:
            X_test = []
                
        scene, qd_class_ids = [], []
        obj_names, sketches = [], []
        sketch_divisions, sketch_bboxes = [0], []
        last_stroke = np.asarray([0, 0])
        
        for sketch_num, sketch in enumerate(scene_info):
            drawing = sketch["drawing"]
            label = sketch["labels"]
            label = label.lower().strip()
            
            if label == "incomplete":
                continue
            label, sel_id = get_cls_info(label, metadata, external_classes_to_idx)
            
            lines = simplify_drawing(drawing)
            sketch = lines_to_strokes(lines)
            
            sketch[0, :2] -= last_stroke
            last_stroke[0] = lines[-1][-1][0]
            last_stroke[1] = lines[-1][-1][1]
            
            scene.extend(sketch.tolist())
            num_sk_strokes = int(np.sum(sketch[..., -1] == 1))
            sketch_divisions.append(sketch_divisions[-1] + num_sk_strokes)
            
            sketch_temp = copy.deepcopy(sketch)
            abs_sketch = relative_to_absolute(sketch_temp)
            min_x, min_y, max_x, max_y = get_absolute_bounds(abs_sketch)
            
            sketch_bboxes.append([min_x, min_y, max_x, max_y])
            
            obj_names.append(label)
            qd_class_ids.append(sel_id)
            sketches.append(sketch_temp)
            
            if hps["save_embeds"]:
                sketch_temp = copy.deepcopy(np.asarray(sketch_temp))
                abs_sketch = relative_to_absolute(sketch_temp)
                
                min_x, min_y, max_x, max_y = get_absolute_bounds(abs_sketch)
                
                scale = hps['scale_factor'] / max([max_x - min_x, max_y - min_y, 1])
                # align the drawing to the top-left corner, to have minimum values of 0.
                abs_sketch[:, 0] -= min_x
                abs_sketch[:, 1] -= min_y
                
                # uniformly scale the drawing, to have a maximum value of 255.
                abs_sketch[:, :2] *= scale
                
                sketch_relative = absolute_to_relative(abs_sketch)
                sketch_normalized = normalize(sketch_relative)
                X_test.append(sketch_normalized)
        
        if len(sketch_bboxes) == 0:
            continue
        
        paths_dict[img_id] = {"coord_img_paths": [], 
                              "sketch_img_paths": [],
                              "vector_img_paths": [],
                              "class_ids": []}
        if hps["save_embeds"]:
            object_embeds, predicted, class_scores = retrieve_embedding_and_classes_from_batch(model, X_test)
            object_embeds = object_embeds.tolist()
            class_scores = class_scores.numpy().tolist()
        else:
            object_embeds = None
            class_scores = None
            predicted = None
            
        sketch_divisions = np.asarray(sketch_divisions)
        scene_strokes = np.asarray(scene)
        
        # scale each sketch object bboxes and add scene sketch bbox
        sketch_bboxes = np.asarray(sketch_bboxes)
        img_h = hps["img_h"]
        img_w = hps["img_w"]
        scene_size = max(img_h, img_w)
        sketch_bboxes = np.insert(sketch_bboxes, 0, [0.0, 0.0, img_w, img_h], axis=0)
        
        orig_coords = copy.deepcopy(sketch_bboxes)
        coords = torch.Tensor(sketch_bboxes)
        scaled_coords = scale_bboxes(coords, img_w, img_h, scene_size)
        coords = torch.Tensor(scaled_coords)
                
        # Coords will be processed as image
        raster_coords = make_coords_img(coords, scene_size)
        
        for idx, class_id in enumerate(qd_class_ids):
            qd_class = obj_names[idx]
            save_name = qd_class + "_" + str(img_id) + "_" + str(idx+1)
                    
            # save sketch coord image
            image_path = os.path.join(images_dir, save_name + '.png')
            coords_img = raster_coords[idx+1].astype(np.uint8)
            cv2.imwrite(image_path, coords_img)
                    
            # save sketch image using stroke-3 format
                    
            sketches_path = os.path.join(sketches_dir, save_name + '.png')
            sketch_temp = np.asarray(sketches[idx])
            sketches_img = draw_sketch(sketch_temp, sketches_path, white_bg=True, max_dim=hps['obj_size'])                
                
            vector_path = os.path.join(vectors_dir, save_name + '.json')
            vector_dict = {"img_w": img_w, "img_h": img_h, "coord": orig_coords[idx+1].tolist(), "stroke": sketch_temp.tolist(),
                           "object_embeds": object_embeds[idx], "class_scores": class_scores[idx], "predicted": predicted[idx]}
            
            with open(vector_path, "w") as f:
                json.dump(vector_dict, f)
                    
                    
            res_dict["data"].append({
                "class_id": class_id,
                "class_name": qd_class,
                "vectors_path": os.path.join('vectors', save_name + '.json'),
                "images_path": os.path.join('images', save_name + '.png'),
                "sketches_path": os.path.join('sketches', save_name + '.png'),
                "scenes_path": os.path.join('scenes', str(img_id) + '.png')
            })
            
            # Save paths info            
            paths_dict[img_id]["class_ids"].append(class_id)
            
            path = os.path.join('sketches', save_name + '.png')
            paths_dict[img_id]["sketch_img_paths"].append(path)
                
            path = os.path.join('images', save_name + '.png')
            paths_dict[img_id]["coord_img_paths"].append(path)
            
            path = os.path.join('vectors', save_name + '.json')
            paths_dict[img_id]["vector_img_paths"].append(path)
            
        os.system(f"mkdir -p {os.path.join(scenes_dir, 'pngs')}")
              
        img_path = os.path.join(scenes_dir, 'pngs', str(img_id) + '.png')
        raster_scene = draw_sketch(copy.deepcopy(scene_strokes), img_path, white_bg=True, keep_size=True, max_dim=scene_size)
        
        image_counter += 1
                                                 
        scene_dict = {"img_id": img_id, 
                      "img_w": img_w, "img_h": img_h, 
                      "scene": raster_scene.tolist(),
                      "object_divisions": sketch_divisions.tolist(),
                      "scene_strokes": scene_strokes.tolist(),
                      "object_embeds": object_embeds,
                      "class_scores": class_scores,
                      "predicted": predicted}
        
        os.system(f"mkdir -p {os.path.join(scenes_dir, 'jsons')}")                
        with open(os.path.join(scenes_dir, 'jsons', str(img_id) + ".json"), "w") as f:
            json.dump(scene_dict, f)
        
        add_meta = {"external_classes_to_idx": external_classes_to_idx}
        with open(os.path.join(target_dir, "external_classes_to_idx.json"), 'w') as outfile:
            json.dump(add_meta, outfile)
    
    with open(os.path.join(save_pth, "data_info.json"), "w") as f:
        json.dump(res_dict, f)
    
    with open(os.path.join(save_pth, "path_info.json"), "w") as f:
        json.dump(paths_dict, f)
        
    return image_counter
            
            
def main():
    # Parsing arguments
    parser = argparse.ArgumentParser(
        description='Prepare large dataset for chunked loading')
    parser.add_argument('--data-filename', default='scene-dataset-merged.json')
    parser.add_argument('--target-dir', default='custom-dataset')
    parser.add_argument('--hparams', type=str)
    
    parser.add_argument('--class-relationship', type=str, default='qd_coco_files/quickdraw_to_coco_v2.json')
    parser.add_argument('--sknet-classes', type=str, default='../Word-Embedding-Extraction/out/sketchnet_meta_updated.json')

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
    qd_classes_to_idx = {c.lower(): i for i, c in enumerate(sketch_to_coco.keys())}
    
    # load the Sknet class ids
    with open(args.sknet_classes) as f:
        sknet_data = json.load(f)
    print("Sketchnet class ids are loaded.")

    metadata = {"sknet_classes_to_idx": sknet_data['sketchnet_classes_to_idx'],
               "qd_classes_to_idx": qd_classes_to_idx, 
               "coco_to_sketch": coco_to_sketch}
    with open(os.path.join(args.target_dir, "metadata.json"), 'w') as outfile:
        json.dump(metadata, outfile)
    print("Metadata is saved.")
    
    test_n_samples = load_custom_data(args.data_filename, args.target_dir, metadata, hps)
    print("Saved {} images for custom test set".format(test_n_samples))


if __name__ == '__main__':
    main()