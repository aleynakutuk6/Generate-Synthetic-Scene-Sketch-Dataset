import glob
import argparse
import json
import os
import sys
import pickle

import imageio
from collections import defaultdict
import skimage.transform as sk_transform
import pycocotools.mask as coco_mask_utils
import numpy as np
import tensorflow as tf
from utils import graph
from utils import bbox
from utils import hparams

def default_hparams():
    hps = hparams.HParams(
        obj_size=480,
        min_object_size=0.05,
        min_objects_per_image=3,
        max_objects_per_image=5,
        max_objects_per_category=3,
        min_category_per_image=3,
        overlap_ratio=0.4,
        image_margin=20,
        include_image_obj=False,
        save_embeds=True,
        excluded_meta_file='qd_coco_files/coco_mats_objs_sketchables_v2.json',
    )
    return hps



def create_vocab(instances_data_categories, stuff_data_categories):
    object_idx_to_name = {}
    object_name_to_idx = {}
    for category_data in instances_data_categories:
        category_id = category_data['id']
        category_name = category_data['name']
        object_idx_to_name[category_id] = category_name
        object_name_to_idx[category_name] = category_id
    for category_data in stuff_data_categories:
        category_id = category_data['id']
        category_name = category_data['name']
        object_idx_to_name[category_id] = category_name
        object_name_to_idx[category_name] = category_id
    return object_idx_to_name, object_name_to_idx


def load_metadata_from_json(instances_json, stuff_json):

    with open(instances_json, 'r') as f:
        instances_data = json.load(f)

    stuff_data = None
    if stuff_json is not None and stuff_json != '':
        with open(stuff_json, 'r') as f:
            stuff_data = json.load(f)
    return instances_data, stuff_data


def create_images_and_objects_dataset(instances_data, stuff_data):
    image_ids, image_id_to_filename, image_id_to_size = [], {}, {}
    image_id_to_objects = defaultdict(list)
    for image_data in instances_data['images']:
        image_id = image_data['id']
        filename = image_data['file_name']
        width = image_data['width']
        height = image_data['height']
        image_ids.append(image_id)
        image_id_to_filename[image_id] = filename
        image_id_to_size[image_id] = (width, height)

    # Add object data from instances
    for object_data in instances_data['annotations']:
        image_id_to_objects[object_data['image_id']].append(object_data)

    # Add object data from stuff
    image_ids_with_stuff = set()
    new_image_ids = set()
    for object_data in stuff_data['annotations']:
        image_ids_with_stuff.add(object_data['image_id'])
        image_id_to_objects[object_data['image_id']].append(object_data)
    total_objs = 0
    for image_id in image_ids:
        num_objs = len(image_id_to_objects[image_id])
        new_image_ids.add(image_id)
        total_objs += num_objs

    image_ids = list(new_image_ids)

    objects_list = set()
    for image_id in image_ids:
        for object in image_id_to_objects[image_id]:
            object_class = object['category_id']
            objects_list.add(object_class)
    return list(objects_list), total_objs, image_ids, image_id_to_filename, image_id_to_size, image_id_to_objects


def should_keep_object(size, obj, hps, meta):
    _, _, w, h = obj['bbox']
    W, H = size
    box_area = (w * h) / (W * H)
    box_ok = box_area > hps['min_object_size']
    object_name = meta['obj_ID_to_name'][str(obj['category_id'])]
    category_ok = object_name in meta['concrete_objs'] or object_name in meta['allowed_materials'] or object_name in meta['fully_excluded_objs']
    is_not_other = object_name != 'other'
    return box_ok and category_ok and is_not_other


def filter_objs_customized(size, objs, hps, meta):
    c_objs, obj_ids = [], []
    bboxes = []
    obj_cats = {}
    W, H = size
    max_inst_cnt = hps['max_objects_per_category']
    max_obj_cnt = 10
    
    for obj in objs:
        if obj['id'] in obj_ids:
            print("Repeated object {} found!".format(obj['id']))
            continue
        if should_keep_object(size, obj, hps, meta):
            obj_id = obj['id']
            obj_ids.append(obj_id)
            x, y, w, h = obj['bbox']
            o_cat_id = obj['category_id']
            box_area = (w * h) / (W * H)
            
            if o_cat_id not in obj_cats.keys():
                obj_cats[o_cat_id] = []
                obj_cats[o_cat_id].append([box_area])
                obj_cats[o_cat_id].append([obj_id])
            else:
                obj_cats[o_cat_id][0].append(box_area)
                obj_cats[o_cat_id][1].append(obj_id)
    
    print("before", obj_cats)
    if len(obj_cats) < hps['min_category_per_image']:
        return c_objs
            
    for cat_id in obj_cats:
        bboxes = np.asarray(obj_cats[cat_id][0])
        obj_ids = obj_cats[cat_id][1]
        sort_index = np.flip(np.argsort(bboxes))
        if len(sort_index) > max_inst_cnt:
            sort_index = sort_index[:max_inst_cnt]
        
        obj_cats[cat_id][0] = np.asarray(obj_cats[cat_id][0])[sort_index]
        obj_cats[cat_id][1] = np.asarray(obj_cats[cat_id][1])[sort_index]
    
    print("obj_cats", obj_cats)
    pushed_obj = True
    n_id = 0
    n_pushed = 0
    sel_ids = set()
    while pushed_obj and n_pushed < max_obj_cnt:
        pushed_obj = False
        for cat_id in obj_cats:
            if len(obj_cats[cat_id][1]) > n_id:
                obj_id = obj_cats[cat_id][1][n_id]
                sel_ids.add(obj_id)
                n_pushed += 1
                pushed_obj = True
            if n_pushed >= max_obj_cnt:
                break
        n_id += 1
    
    obj_ids = []
    for obj in objs:
        obj_id = obj['id']
        if obj_id in sel_ids:
            c_objs.append(obj)
            obj_ids.append(obj_id)
    print("obj ids:", obj_ids)
    return c_objs
    

def prepare_and_load_metadata(instances_json, stuff_json):

    # load the metadata
    instances_data, stuff_data = load_metadata_from_json(instances_json, stuff_json)

    # separate category metadata
    object_idx_to_name, object_name_to_idx = create_vocab(
        instances_data['categories'], stuff_data['categories'])

    # break down data by image sample
    results = create_images_and_objects_dataset(
        instances_data, stuff_data)
    (objects_list, total_objs, image_ids, image_id_to_filename,
        image_id_to_size, image_id_to_objects) = results

    # category labels start at 1, so use 0 for __image__
    object_name_to_idx['__image__'] = 0
    object_idx_to_name[0] = '__image__'

    return (object_idx_to_name, object_name_to_idx, objects_list, total_objs,
            image_ids, image_id_to_filename,
            image_id_to_size, image_id_to_objects)

    
def load_data_given_imageid(image_id, image_id_to_objects, image_id_to_size, hps, meta):
    
    objs = image_id_to_objects[image_id]
    size = image_id_to_size[image_id]
    c_objs = filter_objs_customized(size, objs, hps, meta)
        
    c_objs_y = []
    for i, object_data in enumerate(c_objs):
        c_objs_y.append(object_data['category_id'])
    
    c_objs_y = np.array(c_objs_y)
    print("c_objs_y", c_objs_y)
    c_objs_y = [meta['obj_ID_to_idx'][y] for y in c_objs_y]
    coco_classes = [meta['obj_idx_to_name'][c] for c in c_objs_y]
    print("coco_classes", coco_classes)


############################################################################


hps = default_hparams()
hps = dict(hps.values())
    
dataset_dir = '/datasets/COCO'

train_instances_json = os.path.join(dataset_dir, 'annotations/instances_train2017.json')
train_stuff_json = os.path.join(dataset_dir, 'annotations/stuff_train2017.json')
val_instances_json = os.path.join(dataset_dir, 'annotations/instances_val2017.json')
val_stuff_json = os.path.join(dataset_dir, 'annotations/stuff_val2017.json')

# load up all train metadata
print("Loading train metadata...")
(object_idx_to_name, object_name_to_idx, objects_list, total_objs,
train_image_ids, _,
image_id_to_size,
image_id_to_objects) = prepare_and_load_metadata(train_instances_json, train_stuff_json)
    
# load up all valid metadata
print("Loading validation metadata...")
(_, _, _, _,
valid_image_ids, _,
valid_image_id_to_size,
valid_image_id_to_objects) = prepare_and_load_metadata(val_instances_json, val_stuff_json)
     
# load up all val and train dicts together
image_id_to_objects.update(valid_image_id_to_objects)
image_id_to_size.update(valid_image_id_to_size)


target_dir = 'coco-records-latest'
meta_filename = os.path.join(target_dir, "meta.json")
with open(meta_filename) as clrf:
    meta = json.load(clrf)
    
image_id = 174567
load_data_given_imageid(image_id, image_id_to_objects, image_id_to_size, hps, meta)




