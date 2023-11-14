import glob
import os
import json
import numpy as np
import nltk
import torch
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from tqdm import tqdm
from nltk.corpus import stopwords
# from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer


def visualize_TSNE(word_embeds_path, sknet_metadata):

    with open(word_embeds_path, "r") as f:
        word_embeds = json.load(f)

    with open(sknet_metadata, "r") as f:
        metadata = json.load(f)

    words = metadata["sketchnet_classes_to_idx"].keys()

    # Extract words and their embeddings
    word_ids = list(word_embeds.keys())
    vectors = np.array([word_embeds[word_id] for word_id in word_ids])

    # Use t-SNE to reduce the dimensions to 2D for visualization
    tsne = TSNE(n_components=2, random_state=42)
    vectors_2d = tsne.fit_transform(vectors)

    # Plot the words in 2D space
    plt.figure(figsize=(15, 12))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c='b')

    # Annotate points with word labels
    for i, word in enumerate(words):
        plt.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]), textcoords='offset points', xytext=(0,10), ha='center')

    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('t-SNE Visualization of Word Embeddings')
    plt.show()
    

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
        

# Extract word embeddings for Quickdraw classes

qd_save_filepath = "json_files/QD_word_embeds_with_transformer.json"
data_path = "coco-records-small/qd_coco_meta.json"
# uniq_classes = get_uniq_classes(data_path, "qd")
# save_word_embeds(uniq_classes, qd_save_filepath)

# Find embed similarity

find_embed_similarity(qd_save_filepath, "qd")
get_similar_ids("json_files/qd_embed_distances.npy", data_path, "qd")


# Extract word embeddings for SketchNet classes

sknet_save_filepath = "json_files/SketchNet_word_embeds_with_transformer.json"
data_path = "../Topic-Modeling/SketchNet/sketchnet_data.json"
meta_save_filename = "json_files/sketchnet_meta.json"
# save_sketchnet_metadata(data_path, meta_save_filename)
# uniq_classes = get_uniq_classes(meta_save_filename, "sknet")
# save_word_embeds(uniq_classes, sknet_save_filepath)

# Find embed similarity

# find_embed_similarity(sknet_save_filepath, "sknet")
# get_similar_ids("json_files/sknet_embed_distances.npy", meta_save_filename, "sknet")
# visualize_TSNE(sknet_save_filepath, meta_save_filename)
