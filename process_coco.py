import argparse
import json
import os
import random
import imageio
import numpy as np
import skimage.transform as sk_transform
import pycocotools.mask as coco_mask_utils

from tqdm import tqdm
from generate_scene import *
from utils import hparams, coco, tfrecord, sketch
from bbox_extractor import *
    
"""
def scale_image(image, image_size):
    scaled = sk_transform.resize(image, (image_size, image_size))
    if scaled.shape[-1] == 3:
        scaled = scaled[..., 0]
    if len(scaled.shape) == 2:
        scaled = np.reshape(scaled,
                            (image_size, image_size, 1))
    return scaled
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
        max_objects_per_image=5,
        max_objects_per_category=3,
        min_category_per_image=3,
        overlap_ratio=0.4,
        image_margin=20,
        include_image_obj=False,
        excluded_meta_file='qd_coco_files/coco_mats_objs_sketchables_v2.json',
    )
    return hps


def load_all_data_and_save_in_chunks(image_ids, image_dirs, id_to_size, id_to_objects, 
                                     base_filepath, class_files, set_type, hps, c_meta, qd_meta):
                               
    image_counter = 0

    for i in tqdm(range(0, len(image_ids))):
        img_id = image_ids[i]                
            
        save_path = os.path.join(base_filepath, str(img_id))
        
        prefix = "0" * (12 - len(str(img_id)))
        filename = prefix + str(img_id) + ".jpg"
        image_dir = image_dirs[i]
        img = imageio.imread(os.path.join(image_dir, filename))
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
    parser.add_argument('--target-dir', default='coco-records-temp')
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
    
    f = open(os.path.join(args.target_dir, "data_info.txt"), "w")
    f.write('############################## Dataset Info ################################ \n')
    f.write("image_size: {} \n".format(hps["image_size"]))
    f.write("min_object_size: {} \n".format(hps["min_object_size"]))
    f.write("min_objects_per_image: {} \n".format(hps["min_objects_per_image"]))
    f.write("max_objects_per_image: {} \n".format(hps["max_objects_per_image"]))
    f.write("max_objects_per_category: {} \n".format(hps["max_objects_per_category"]))
    f.write("min_category_per_image: {} \n".format(hps["min_category_per_image"]))
    f.write("overlap_ratio: {} \n".format(hps["overlap_ratio"]))
    f.write("mask_size: {} \n".format(hps["mask_size"]))
    f.write("image_margin: {} \n".format(hps["image_margin"]))
    f.write("include_image_obj: {} \n".format(hps["include_image_obj"]))
    f.write("excluded_meta_file: {} \n".format(hps["excluded_meta_file"]))
    f.close()

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
     train_image_ids, _,
     image_id_to_size,
     image_id_to_objects) = coco.prepare_and_load_metadata(train_instances_json, train_stuff_json)
    
    # load up all valid metadata
    print("Loading validation metadata...")
    (_, _, _, _,
     valid_image_ids, _,
     valid_image_id_to_size,
     valid_image_id_to_objects) = coco.prepare_and_load_metadata(val_instances_json, val_stuff_json)
     
    # load up all val and train dicts together
    image_id_to_objects.update(valid_image_id_to_objects)
    image_id_to_size.update(valid_image_id_to_size)
    
    # Image dirs
    train_image_dirs = [train_image_dir] * len(train_image_ids)
    val_image_dirs = [val_image_dir] * len(valid_image_ids)
    all_dirs = train_image_dirs + val_image_dirs
    
    # Image ids
    all_image_ids = train_image_ids + valid_image_ids

    # Shuffle both in the same order
    temp = list(zip(all_image_ids, all_dirs))
    random.shuffle(temp)
    res1, res2 = zip(*temp)
    all_image_ids, all_dirs = list(res1), list(res2)
    
    # Divide train - val - test datasets with split ratio 60 - 10 - 30
    train_size = int(len(all_image_ids) * 60 / 100)
    val_size = int(len(all_image_ids) * 10 / 100)
    test_size = len(all_image_ids) - (train_size + val_size)
    
    train_image_ids = all_image_ids[:train_size]
    valid_image_ids = all_image_ids[train_size:train_size+val_size]
    test_image_ids = all_image_ids[train_size+val_size:]
    
    train_image_dirs = all_dirs[:train_size]
    val_image_dirs = all_dirs[train_size:train_size+val_size]
    test_image_dirs = all_dirs[train_size+val_size:]
    
    test_n_images, valid_n_images, train_n_images = len(test_image_ids), len(valid_image_ids), len(train_image_ids)
    
    f = open(os.path.join(args.target_dir, "data_info.txt"), "a")
    f.write("train_n_images: {} \n".format(train_n_images))
    f.write("valid_n_images: {} \n".format(valid_n_images))
    f.write("test_n_images: {} \n".format(test_n_images))
    f.close()

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
        image_ids=valid_image_ids,
        image_dirs=val_image_dirs,
        id_to_size=image_id_to_size,
        id_to_objects=image_id_to_objects,
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
        image_ids=test_image_ids,
        image_dirs=test_image_dirs,
        id_to_size=image_id_to_size,
        id_to_objects=image_id_to_objects,
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
        image_ids=train_image_ids,
        image_dirs=train_image_dirs,
        id_to_size=image_id_to_size,
        id_to_objects=image_id_to_objects,
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