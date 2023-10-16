import skimage.io
import glob
import os
import json
import numpy as np
import nltk
import torch

from scipy.io import loadmat
from scipy.stats import mode
from PIL import Image
from nltk.stem import WordNetLemmatizer
from pycocotools.coco import COCO
from tqdm import tqdm
from gensim.models import Word2Vec

nltk.download('omw-1.4')
nltk.download('wordnet')

def get_coco_classes(data_dir):
    
    lemmatizer = WordNetLemmatizer()
    
    ## Get classes for coco
    coco_classes = {} # id --> category
    register = open(os.path.join(data_dir, 'coco_stuff_labels.txt'), 'r').readlines()
    for row in register:
        row = [item.strip() for item in row.split(':')]
        coco_classes[int(row[0])] = lemmatizer.lemmatize(row[1])
    return coco_classes
        
def process_SketchyCOCO(data_dir, coco_classes):
    
    num_obj_sketchycoco = []
    sketchycoco_classes = {} # id --> category
    print ('Processing SketchyCOCO. Please wait ...')
    
    sketchycoco_2_coco = {1:2, 2:3, 3:4, 4:5, 5:10, 6:11, 7:17, 8:18, 9:19, 10:20,\
        11:21, 12:22, 14:24, 15:25, 17:124, 18:106, 19:169}
    
    list_all_sketches = glob.glob(os.path.join(data_dir, 'Annotation', 'paper_version', '*', 'INSTANCE_GT', '*.mat'))
    for sketch_path in tqdm(list_all_sketches):
        classes = np.unique(loadmat(sketch_path)['INSTANCE_GT'])
        num_obj_sketchycoco.append(len(classes))
        for key in classes:
            if key not in sketchycoco_2_coco.keys():
                continue
            
            coco_id = sketchycoco_2_coco[key]
            category = coco_classes[coco_id]
            if category in sketchycoco_classes.keys():
                sketchycoco_classes[category] += 1
            else:
                sketchycoco_classes[category] = 1
                
    print ('SketchyCOCO => Mean: {}, SD: {}, Max: {}, Min: {}'.format(
        np.mean(num_obj_sketchycoco), np.std(num_obj_sketchycoco),
        np.max(num_obj_sketchycoco), np.min(num_obj_sketchycoco)))
    
    return sketchycoco_classes


def get_img_ids(data_dir):
    
    print('Reading SketchyCOCO...')
    all_img_ids = [os.path.split(item)[-1][:-4] for item in glob.glob(os.path.join(data_dir, 'Annotation', 'paper_version', '*', 'reference_image', '*.jpg'))]
    return all_img_ids


def save_coco_embeds(data_dir, coco_classes, out_path, word_embed_dim=256):

    sketchycoco_2_coco = {1:2, 2:3, 3:4, 4:5, 5:10, 6:11, 7:17, 8:18, 9:19, 10:20,\
        11:21, 12:22, 14:24, 15:25, 17:124, 18:106, 19:169}
        
    uniq_classes = set()
    coco_word_embeds = {}
    coco_cls_to_id = {}
    
    list_all_sketches = glob.glob(os.path.join(data_dir, 'Annotation', 'paper_version', '*', 'INSTANCE_GT', '*.mat'))
    for sketch_path in tqdm(list_all_sketches):
        classes = np.unique(loadmat(sketch_path)['INSTANCE_GT'])
        for key in classes:
            if key not in sketchycoco_2_coco.keys():
                continue
                
            coco_id = sketchycoco_2_coco[key]
            coco_cls = coco_classes[coco_id]
            coco_cls_to_id[coco_cls] = key
            uniq_classes.add(coco_cls)
            
    labels = [[c] for c in list(uniq_classes)]
    model = Word2Vec(labels, vector_size=word_embed_dim, min_count=1)
    
    for word in labels:
        word_str = word[0]
        coco_word_embeds[str(coco_cls_to_id[word_str])] = model.wv[word_str].astype(float).tolist()
    
    with open(os.path.join(out_path, "coco_word_embeds.json"), "w") as f:
        json.dump(coco_word_embeds, f)
        

def save_QD_embeds(data_dir, out_path, filename, word_embed_dim=100):

    with open(os.path.join(data_dir, "qd_coco_meta.json")) as f:
        qd_metadata = json.load(f)
    
    qd_classes_to_idx = qd_metadata['qd_classes_to_idx']
    uniq_classes = qd_classes_to_idx.keys()
    
    QD_word_embeds = {}
            
    labels = [[c.lower()] for c in list(uniq_classes)]
    model = Word2Vec(labels, vector_size=word_embed_dim, min_count=1)
    
    for i, word in enumerate(labels):
        word_str = word[0]
        QD_word_embeds[str(i)] = model.wv[word_str].astype(float).tolist()
    
    with open(os.path.join(out_path, filename), "w") as f:
        json.dump(QD_word_embeds, f)


def find_embed_similarity(QD_word_embeds, encs, top_k=10):
    
    embs_mtx, embs_keys_dict = [], {}
    for k_idx, k in enumerate(QD_word_embeds):
        embs_keys_dict[k_idx] = k
        embs_mtx.append(QD_word_embeds[k])
    embs_mtx = torch.Tensor(embs_mtx).to("cpu") 
    
    diff = encs.unsqueeze(0) - embs_mtx.unsqueeze(1)
    diff = diff.pow(2).sum(dim=-1).sqrt() 
        
    embed_inds = diff.argsort(dim=0)
    embed_inds = embed_inds.cpu().permute(1, 0).numpy()[:, :top_k].tolist()
    
    return embed_inds
    

def get_bbox(data_dir, img_id):

    save_dir = os.path.join('images', str(img_id))
    print("Processing img {}".format(str(img_id)))

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
   
    instance_gt = glob.glob(os.path.join(data_dir, 'Annotation', 'paper_version', '*', 'INSTANCE_GT', f'{img_id}.mat'))
    bbox_gt = glob.glob(os.path.join(data_dir, 'Annotation', 'paper_version', '*', 'BBOX', f'{img_id}.mat'))
    class_gt = glob.glob(os.path.join(data_dir, 'Annotation', 'paper_version', '*', 'CLASS_GT', f'{img_id}.mat'))
    sketches_gt = glob.glob(os.path.join(data_dir, 'Sketch', 'paper_version', '*', f'{img_id}.png'))
    images_gt = glob.glob(os.path.join(data_dir, 'Annotation', 'paper_version', '*', 'reference_image', f'{img_id}.jpg'))
    
    for i_gt in instance_gt:
        if os.path.isfile(i_gt):
            instance_obj = loadmat(i_gt)['INSTANCE_GT']
            break
    
    for bbx_gt in bbox_gt:
        if os.path.isfile(bbx_gt):
            bbox_obj = loadmat(bbx_gt)['BBOX']
            
    for cls_gt in class_gt:
        if os.path.isfile(cls_gt):        
            class_obj = loadmat(cls_gt)['CLASS_GT']
            print("cls obj", class_obj)
    
    for img_gt in images_gt:
        if os.path.isfile(img_gt):
            im = Image.open(img_gt)
    
    for sketch_gt in sketches_gt:
        if os.path.isfile(sketch_gt):
            sketch_img = Image.open(sketch_gt)
    
    instances = np.unique(instance_obj)
        
    f = open(os.path.join('labels', str(img_id) + '.txt'), "w")
    
    sketch_img.save(os.path.join(save_dir, f'0.png'))
    im.save(os.path.join(save_dir, f'image.png'))
    i_x_max, i_y_max = im.size # width, height
    i_x_min, i_y_min = 0, 0
    img_area = (i_x_max - i_x_min) * (i_y_max - i_y_min)
    f.write(f'0,{i_x_min},{i_y_min},{i_x_max},{i_y_max},{img_area},-1\n')
    
    for instance in instances:
        x_coords, y_coords = np.where(instance_obj == instance)
        print(im.size, x_coords, y_coords)
        x_min = max(min(x_coords), 0)
        x_max = min(max(x_coords), i_x_max)
        y_min = max(min(y_coords), 0)
        y_max = min(max(y_coords), i_y_max)
        img_area = (x_max - x_min) * (y_max - y_min)
        
        cls = mode(class_obj[x_coords[:10], y_coords[:10]], keepdims=False)
        found_cls = cls.mode
        if found_cls == 16 or found_cls == 0:
            continue
          
        f.write(f'{instance},{x_min},{y_min},{x_max},{y_max},{img_area},{found_cls}\n')
        print(i_x_min, x_min, i_y_min, y_min, i_x_max, x_max, i_y_max, y_max)
        sketch_img_crop = sketch_img.crop((x_min, y_min, x_max, y_max))
        sketch_img_crop.save(os.path.join(save_dir, f'{instance}.png'))
        
    f.close()
        
        
def read_COCO(data_dir, dataType='train2017'):
    
    annFile = '%s/annotations/stuff_%s.json' % (data_dir, dataType)
    cocoGt = COCO(annFile)
    
    # Display COCO stuff categories and supercategories
    categories = cocoGt.loadCats(cocoGt.getCatIds())
    categoryNames = [cat['name'] for cat in categories]
    print('COCO Stuff leaf categories:')
    print(categoryNames)

    superCategoryNames = sorted(set([cat['supercategory'] for cat in categories]))
    print('COCO Stuff super categories:')
    print(superCategoryNames)

    # Load info for a random image
    imgIds = cocoGt.getImgIds()
    imgId = imgIds[np.random.randint(0, len(imgIds))]
    img = cocoGt.loadImgs(imgId)[0]
    img.save('img.png', quality=95)

    # Load and display image
    I = skimage.io.imread(img['coco_url'])
    # print(I.size)

    # Load and display stuff annotations
    annIds = cocoGt.getAnnIds(imgIds=img['id'])
    anns = cocoGt.loadAnns(annIds)


word_embed_dim = 100
out_path = "json_files"
filename = f"QD_word_embeds_dim_{word_embed_dim}.json"

# save_QD_embeds("coco-records", out_path, filename)

with open(os.path.join(out_path, filename), "r") as f:
    word_embeds = json.load(f)

encs = torch.Tensor(word_embeds["328"])
embed_inds = find_embed_similarity(word_embeds, encs)
print("embed inds:", embed_inds)

"""
data_dir = "/datasets/SketchyCOCO/Scene"
coco_classes = get_coco_classes('coco_stuff')
save_coco_embeds(data_dir, coco_classes, "json_files")

img_ids = get_img_ids(data_dir)

for img_id in tqdm(img_ids[:1]):
    get_bbox(data_dir, img_id)
"""
    