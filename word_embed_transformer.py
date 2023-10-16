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


def visualize_TSNE(word_embeds):
    # Extract words and their embeddings
    words = list(word_embeds.keys())
    vectors = np.array([word_embeds[word] for word in words])

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


def save_QD_embeds(data_dir, out_path, filename):
    
    with open(os.path.join(data_dir, "qd_coco_meta.json")) as f:
        qd_metadata = json.load(f)

    qd_classes_to_idx = qd_metadata['qd_classes_to_idx']
    uniq_classes = qd_classes_to_idx.keys()

    QD_word_embeds = {}
    # texts = get_txt_data(os.path.join(out_path, txt_filename))
    
    labels = preprocess(uniq_classes)
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    # embedding = model.encode(sentence)

    # model = Word2Vec(sentences=texts, vector_size=100, window=5, min_count=1, workers=4)
    # model = Word2Vec(labels, vector_size=word_embed_dim, min_count=1)
    # model.train(labels, total_examples=1, epochs=1)

    for i, word in enumerate(labels):
        QD_word_embeds[str(i)] = model.encode(word).astype(float).tolist()
        # QD_word_embeds[word_str] = model.wv[word_str].astype(float).tolist()

    with open(os.path.join(out_path, filename), "w") as f:
        json.dump(QD_word_embeds, f)
        

out_path = "json_files"
filename = "QD_word_embeds_with_transformer.json"

save_QD_embeds("coco-records", out_path, filename)

"""
with open(os.path.join(out_path, filename), "r") as f:
    word_embeds = json.load(f)

encs = torch.Tensor(word_embeds["mona lisa"])
embed_inds = find_embed_similarity(word_embeds, encs)
print("embed inds:", embed_inds)
"""
