import argparse
import json
import os
import random

import imageio
from skimage import color
from collections import defaultdict
from tqdm import tqdm
import skimage.transform as sk_transform
import pycocotools.mask as coco_mask_utils
import numpy as np

from generate_scene import *
from utils import hparams, coco, tfrecord, sketch
    

def scale_image(image, image_size):
    scaled = sk_transform.resize(image, (image_size, image_size))
    if scaled.shape[-1] == 3:
        scaled = scaled[..., 0]
    if len(scaled.shape) == 2:
        scaled = np.reshape(scaled,
                            (image_size, image_size, 1))
    return scaled
    
    
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


def find_closest_qd_obj(o_width, o_height, set_type, npz_name, k=1):
    
    ratio_data = np.load(os.path.join("qd_ratios", set_type, npz_name, "ratios.npy"))
    
    obj_ratio = o_width / o_height
    diff = np.absolute(obj_ratio - ratio_data)
    top_idxs = np.argsort(diff)[:k]
    
    rand_id = random.randint(0, len(top_idxs)-1)
    sel_sample_id = int(top_idxs[rand_id])
    
    return sel_sample_id  
    

def default_hparams():
    hps = hparams.HParams(
        image_size=224,
        min_object_size=0.05,
        min_objects_per_image=3,
        max_objects_per_image=20,
        max_objects_per_category=3,
        min_category_per_image=2,
        overlap_ratio=0.4,
        mask_size=64,
        image_margin=20,
        include_image_obj=False,
        excluded_meta_file='qd_coco_files/coco_mats_objs_sketchables_v2.json',
    )
    return hps


def load_all_data_and_save_in_chunks(n_chunks, chunk_size, image_ids,
                                     image_dir, id_to_filename, id_to_size, id_to_objects, 
                                     base_filepath, class_files, set_type, hps, c_meta, qd_meta):
                                     
    
                               
    image_counter = 0
    for cur_chunk in range(0, n_chunks):
        start_id = cur_chunk * chunk_size
        end_id = cur_chunk * chunk_size + chunk_size if cur_chunk + chunk_size < len(image_ids) else len(image_ids)

        for i in tqdm(range(start_id, end_id)):
            img_id = image_ids[i]                
            
            save_path = os.path.join(base_filepath, str(img_id))
            """
            if os.path.isdir(save_path) and os.path.isfile(os.path.join(save_path, "data_info.json")):
                continue
            """
            img = imageio.imread(os.path.join(image_dir, id_to_filename[img_id]))
            size = id_to_size[img_id]
            img_w, img_h = size
            objs = id_to_objects[img_id]
            img, c_objs, c_boxes, c_boxes_orig = coco.preprocess(img, size, objs, hps, c_meta)
                
            if img is not None:
                
                save_path = os.path.join(base_filepath, str(img_id))
                
                if not os.path.isdir(save_path):
                    os.mkdir(save_path)
                
                scene_img = {"data": {"image_id": img_id, "objects": []}, "label": 1}
                coco_classes = [c_meta['obj_idx_to_name'][c] for c in c_objs]
                qd_class_ids, sample_ids = [], []
                raster_sketches = np.zeros((len(coco_classes), hps['image_size'], hps['image_size'], 1))
                
                for i, coco_class in enumerate(coco_classes):
                    qd_classes = qd_meta['coco_to_sketch'][coco_class]
                    rand_id = random.randint(0, len(qd_classes)-1)
                    qd_class = qd_classes[rand_id]
                    qd_class_ids.append(qd_meta['qd_classes_to_idx'][qd_class])
                    
                    # load qd data
                    
                    # load qd w/h ratios
                    x, y, w, h = c_boxes_orig[i]
                    sel_id = find_closest_qd_obj(w, h, set_type, qd_class, k=20)
                    
                    qd_data = read_quickdraw_npz(class_files[qd_class], partition=set_type, idx=sel_id)
                    save_dir = os.path.join(save_path, f'{str(i+1)}_{qd_class}.png')
                    raster_sketches[i] = draw_sketch(np.asarray(qd_data), save_dir, white_bg=True)
                    sample_ids.append(sel_id)
                    
                    scene_img["data"]["objects"].append({"stroke-3": np.asarray(qd_data), "x": x, "y": y, "h": h, "w": w})
                        
                # scene info added 
                generated_scene = generate_scene_from_single_img(scene_img)
                sketch = np.asarray(generated_scene["scene_strokes"])
                save_dir = os.path.join(save_path, '0_scene.png')
                scene = draw_sketch(sketch, save_dir, is_absolute=True, white_bg=True)
                
                
                c_boxes_orig = np.insert(c_boxes_orig, 0, [0, 0, img_w, img_h], axis=0)
                
                # object_divisions of scene
                object_divisions = np.asarray(generated_scene["object_divisions"])
                
                # scale each sketch object bboxes and add scene sketch bbox
                sketch_bboxes = np.asarray(generated_scene["sketch_bboxes"])
                sketch_bboxes = scale_bboxes(sketch_bboxes, hps["image_size"], hps["image_margin"])
                scene_min, scene_max = hps["image_margin"], hps["image_size"] - hps["image_margin"]
                sketch_bboxes = np.insert(sketch_bboxes, 0, [scene_min, scene_min, scene_max, scene_max], axis=0)
                
                res_dict = {"img_id": img_id, "sketch_bboxes": sketch_bboxes.tolist(),
                            "raster_sketches": raster_sketches.tolist(), "qd_class_ids": qd_class_ids, 
                            "sample_ids": sample_ids, "scene": scene.tolist(),
                            "object_divisions": object_divisions.tolist()}
                            
                with open(os.path.join(save_path, "data_info.json"), "w") as f:
                    json.dump(res_dict, f)
                    
                image_counter += 1
        
    return image_counter
    

def main():
    # Parsing arguments
    parser = argparse.ArgumentParser(
        description='Prepare large dataset for chunked loading')
    parser.add_argument('--dataset-dir', default='/datasets/COCO')
    parser.add_argument('--target-dir', default='coco-records')
    parser.add_argument('--n-chunks', type=int, default=5)
    parser.add_argument('--test-n-chunks', type=int, default=1)
    parser.add_argument('--valid-n-chunks', type=int, default=1)
    parser.add_argument('--val-size', type=int, default=1024)
    parser.add_argument('--hparams', type=str)
    
    parser.add_argument('--qd-dataset-dir', default='/datasets/quickdraw/sketchrnn/npz')
    parser.add_argument('--class-relationship', type=str, default='qd_coco_files/quickdraw_to_coco_v2.json')

    args = parser.parse_args()
    
    if not os.path.isdir(args.target_dir):
        os.mkdir(args.target_dir)
    
    # load the mapping QD & COCO
    with open(args.class_relationship) as clrf:
        sketch_to_coco = json.load(clrf)
        
    class_files, coco_to_sketch, sketch_to_coco_clean = {}, {}, {}
    for class_name, mapped in sketch_to_coco.items():
        if mapped is not None:
            class_files[class_name] = "{}/{}.npz".format(args.qd_dataset_dir, class_name)
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
        

    hps = default_hparams()
    if args.hparams is not None:
        hps = hps.parse(args.hparams)
    hps = dict(hps.values())

    # get all the full paths
    train_image_dir = os.path.join(args.dataset_dir, 'train2017')
    val_image_dir = os.path.join(args.dataset_dir, 'val2017')
    train_instances_json = os.path.join(args.dataset_dir, 'annotations/instances_train2017.json')
    train_stuff_json = os.path.join(args.dataset_dir, 'annotations/stuff_train2017.json')
    val_instances_json = os.path.join(args.dataset_dir, 'annotations/instances_val2017.json')
    val_stuff_json = os.path.join(args.dataset_dir, 'annotations/stuff_val2017.json')
    train_basename = os.path.join(args.target_dir, "train")
    valid_basename = os.path.join(args.target_dir, "valid")
    test_basename = os.path.join(args.target_dir, "test")
    meta_filename = os.path.join(args.target_dir, "meta.json")
    
    if not os.path.isdir(train_basename):
        os.mkdir(train_basename)
                    
    if not os.path.isdir(valid_basename):
        os.mkdir(valid_basename)
    
    if not os.path.isdir(test_basename):
        os.mkdir(test_basename)

    # load up all train metadata
    print("Loading train metadata...")
    (object_idx_to_name, object_name_to_idx, objects_list, total_objs,
     train_image_ids,
     train_image_id_to_filename,
     train_image_id_to_size,
     train_image_id_to_objects) = coco.prepare_and_load_metadata(train_instances_json, train_stuff_json)
    train_n_images = len(train_image_ids)

    # load up all valid metadata
    print("Loading validation metadata...")
    (_, _, _, _,
     valid_image_ids,
     valid_image_id_to_filename,
     valid_image_id_to_size,
     valid_image_id_to_objects) = coco.prepare_and_load_metadata(val_instances_json, val_stuff_json)

    # break valid and train into two sets
    test_image_ids = valid_image_ids[args.val_size:]
    valid_image_ids = valid_image_ids[:args.val_size]
    test_n_images, valid_n_images = len(test_image_ids), len(valid_image_ids)

    with open(hps['excluded_meta_file']) as emf:
        materials_metadata = json.load(emf)
        concrete_objs = materials_metadata["objects"]
        allowed_materials = materials_metadata["materials"]
        fully_excluded_objs = materials_metadata["fully_excluded"]

    object_id_to_idx = {ident: i for i, ident in enumerate(objects_list)}

    # include info about the extra __image__ object
    object_id_to_idx[0] = len(objects_list)
    objects_list = np.append(objects_list, 0)
    

    print("Saving metadata...")
    PREDICATES_VALUES = ['left of', 'right of', 'above', 'below', 'inside', 'surrounding']
    pred_idx_to_name = ['__in_image__'] + PREDICATES_VALUES
    pred_name_to_idx = {name: idx for idx, name in enumerate(pred_idx_to_name)}
    c_meta = {
        'obj_name_to_ID': object_name_to_idx,
        'obj_ID_to_name': object_idx_to_name,
        'obj_idx_to_ID': objects_list.tolist(),
        'obj_ID_to_idx': object_id_to_idx,
        'obj_idx_to_name': [object_idx_to_name[objects_list[i]] for i in range(len(objects_list))],
        'train_total_objs': total_objs,
        'n_train_samples': train_n_images,
        'n_valid_samples': valid_n_images,
        'n_test_samples': test_n_images,
        'concrete_objs': concrete_objs,
        'allowed_materials': allowed_materials,
        'fully_excluded_objs': fully_excluded_objs,
        'pred_idx_to_name': pred_idx_to_name,
        'pred_name_to_idx': pred_name_to_idx
    }
    with open(meta_filename, 'w') as outfile:
        json.dump(c_meta, outfile)
    
    """
    # validation
    c_meta['n_valid_samples'] = load_all_data_and_save_in_chunks(
        n_chunks=args.valid_n_chunks,
        chunk_size=valid_n_images // args.valid_n_chunks,
        image_ids=valid_image_ids,
        image_dir=val_image_dir,
        id_to_filename=valid_image_id_to_filename,
        id_to_size=valid_image_id_to_size,
        id_to_objects=valid_image_id_to_objects,
        base_filepath=valid_basename,
        class_files=class_files,
        set_type='valid',
        c_meta=c_meta, 
        qd_meta=qd_meta,
        hps=hps)
    print("Saved {} images for valid set".format(c_meta['n_valid_samples']))
    with open(meta_filename, 'w') as outfile:
        json.dump(c_meta, outfile)
        
    """
    # test
    c_meta['n_test_samples'] = load_all_data_and_save_in_chunks(
        n_chunks=args.test_n_chunks,
        chunk_size=test_n_images // args.test_n_chunks,
        image_ids=test_image_ids,
        image_dir=val_image_dir,
        id_to_filename=valid_image_id_to_filename,
        id_to_size=valid_image_id_to_size,
        id_to_objects=valid_image_id_to_objects,
        base_filepath=test_basename,
        class_files=class_files,
        set_type='test',
        c_meta=c_meta, 
        qd_meta=qd_meta,
        hps=hps)
    print("Saved {} images for test set".format(c_meta['n_test_samples']))
    with open(meta_filename, 'w') as outfile:
        json.dump(c_meta, outfile)
    """
    # finally, the train set
    c_meta['n_train_samples'] = load_all_data_and_save_in_chunks(
        n_chunks=args.n_chunks,
        chunk_size=train_n_images // args.n_chunks,
        image_ids=train_image_ids,
        image_dir=train_image_dir,
        id_to_filename=train_image_id_to_filename,
        id_to_size=train_image_id_to_size,
        id_to_objects=train_image_id_to_objects,
        base_filepath=train_basename,
        class_files=class_files,
        set_type='train',
        c_meta=c_meta, 
        qd_meta=qd_meta,
        hps=hps)
    print("Saved {} images for train set".format(c_meta['n_train_samples']))
    with open(meta_filename, 'w') as outfile:
        json.dump(c_meta, outfile)
    """
    
if __name__ == '__main__':
    main()