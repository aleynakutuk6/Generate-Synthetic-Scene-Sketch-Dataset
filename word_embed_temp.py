import glob
import os
import json
import numpy as np
import nltk
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
    

def find_embed_similarity(embed_path, data_type="qd"):
    
    with open(embed_path, "r") as f:
        word_embeds = json.load(f)
    
    embeds_len = len(word_embeds)

    embs_mtx, embs_keys_dict = [], {}
    for k_idx, k in enumerate(word_embeds):
        embs_keys_dict[k_idx] = k
        embs_mtx.append(word_embeds[k])
    embs_mtx = torch.Tensor(embs_mtx).to("cpu")
    
    distances = np.zeros((embeds_len, embeds_len))
    
    for i in range(0, embeds_len):
        for j in range(0, embeds_len): 
            diff = embs_mtx[i] - embs_mtx[j]
            diff = diff.pow(2).sum().sqrt()
            distances[i][j] = diff
    
    np.save(f"json_files/{data_type}_embed_distances.npy", distances)


def get_similar_ids(distances_path, sknet_metadata, data_type="qd", top_k=10):

    with open(sknet_metadata, "r") as f:
        metadata = json.load(f)
    
    if data_type == "qd":
        keyname = 'qd_classes_to_idx'
    elif data_type == "sknet":
        keyname = 'sketchnet_classes_to_idx'
        
    words = list(metadata[keyname].keys())
    data = np.load(distances_path)
    embed_inds = data.argsort(axis=1)
    embed_inds = embed_inds[:, :top_k].tolist()

    save_dict = {}
    for i in range(0, len(data)):
        save_dict[words[i]] = []
        for k in range(1, top_k):
            save_dict[words[i]].append(words[embed_inds[i][k]])

    with open(f"json_files/{data_type}_embed_distances.json", "w") as f:
        json.dump(save_dict, f)
    

def preprocess(uniq_classes):
    stop_words = set(stopwords.words('english'))
    stop_words.add('the')

    labels = []
    for cls in list(uniq_classes):
        new_cls = []
        for token in cls.split(" "):
            token = token.lower()
            if token in stop_words:
                continue
            if len(token) > 2:
                new_cls.append(token)
        labels.append(" ".join(new_cls))
    return labels


def save_word_embeds(uniq_classes, save_filepath):

    QD_word_embeds = {}
    
    labels = preprocess(uniq_classes)
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    for i, word in enumerate(labels):
        QD_word_embeds[str(i)] = model.encode(word).astype(float).tolist()

    with open(save_filepath, "w") as f:
        json.dump(QD_word_embeds, f)
        
        
def get_uniq_classes(data_dir, data_type="qd"):
    
    if data_type == "qd":
        keyname = 'qd_classes_to_idx'
    elif data_type == "sknet":
        keyname = 'sketchnet_classes_to_idx'
        
    with open(data_dir) as f:
        data = json.load(f)
    
    classes_to_idx = data[keyname]
    uniq_classes = classes_to_idx.keys()
    
    return uniq_classes
    

def save_sketchnet_metadata(data_dir, save_filename):
    
    with open(data_dir) as f:
        data = json.load(f)
    
    uniq_classes = set()
    for key, item in data.items():
        for word in item["words"]:
            uniq_classes.add(word)
            
    uniq_classes = list(uniq_classes)
    
    mapping_dict = {}
    mapping_dict['sketchnet_classes_to_idx'] = {}
    
    for i, cls in enumerate(uniq_classes):
        mapping_dict['sketchnet_classes_to_idx'][cls] = i
        
    with open(save_filename, "w") as f:
        json.dump(mapping_dict, f)


# Extract word embeddings for SketchNet classes

sknet_save_filepath = "SketchNet_word_embeds_with_transformer.json"
data_path = "sketchnet_data.json"
meta_save_filename = "sketchnet_meta.json"
save_sketchnet_metadata(data_path, meta_save_filename)
uniq_classes = get_uniq_classes(meta_save_filename, "sknet")
save_word_embeds(uniq_classes, sknet_save_filepath)

# Find embed similarity

find_embed_similarity(sknet_save_filepath, "sknet")
get_similar_ids("sknet_embed_distances.npy", meta_save_filename, "sknet")
