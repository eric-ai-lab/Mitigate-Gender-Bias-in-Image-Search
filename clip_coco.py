import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import clip
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import requests
import os
from typing import Set
from utils import mutual_information_2d
from dataset import PrecompDataset
from torchvision.datasets import CocoCaptions
from torchvision.transforms import ToTensor
import time

DATA_PATH = 'data'
DATA_NAME = 'coco_precomp'

begin = time.time()

device = "cuda" if torch.cuda.is_available() else "cpu"
model, transform = clip.load("ViT-B/32", device=device)
dataset = CocoCaptions(root='data/COCO/val2014', annFile = 'data/COCO/annotations/captions_val2014.json')

def open_word_bank(filename: str) -> Set:
    """
    Load the gendered word list.
    """
    word_bank = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            tokens = clip.tokenize(line.strip())[0]
            tokens = tokens[torch.nonzero(tokens)].squeeze().tolist()
            word_bank.append(tokens[1:-1][0])
    return set(word_bank)

MALE_WORD_BANK = open_word_bank('male_word_bank.txt')
FEMALE_WORD_BANK = open_word_bank('female_word_bank.txt')

images = []
captions = []

ids = []
with open('coco_precomp/testall_ids.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        i = int(line.strip())
        if i not in ids:
            ids.append(i)

with open('coco_testall_caps.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        caption = line.strip()
        captions.append(clip.tokenize(caption).to(device))

captions = captions[::5]
count = 0
for img_id in ids:
    coco = dataset.coco
    path = coco.loadImgs(img_id)[0]['file_name']
    image = Image.open(os.path.join(dataset.root, path)).convert('RGB')
    images.append(transform(image).to(device))
    count += 1

# the gender labels of images
# 0 represents gender-neutral
# 1 represents male
# 2 represents female
with open('data/gender_coco.npy', 'rb') as f:
    gender = np.load(f)

def evaluate(similarity: np.ndarray):
    """
    Evaluate the recall and bias performance
    Input:
        similarity: A numpy array of shape [N, N].
    Output:
        recall: Recall@1, Recall@5, Recall@10
        bias:   Bias@[1...10]
    """
    npt = similarity.shape[0]
    ranks = np.zeros(npt)
    male = np.zeros(npt)
    female = np.zeros(npt)

    for i in range(npt):
        inds = np.argsort(similarity[i])[::-1]
        ranks[i] = np.where(inds == i)[0][0]

    # recall
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # bias
    biases = []
    for k in range(1, 11):
        for i in range(npt):
            inds = np.argsort(similarity[i])[::-1]
            inds = inds[:k]
            male[i] = (gender[inds] == 1).sum()
            female[i] = (gender[inds] == 2).sum()

        bias = (male - female) / (male + female + 1e-12)
        bias = bias.mean()
        biases.append(bias)

    return (r1, r5, r10), biases

images   = torch.stack(images)
text_input = torch.zeros(len(captions), model.context_length, dtype=torch.long)
for i, caption in enumerate(captions):
    text_input[i, :len(caption[0])] = caption[0]
text_input = text_input.to(device)

with torch.no_grad():
    image_features = model.encode_image(images).float()
    text_features  = model.encode_text(text_input).float()

# estimate mutula information
mis = []
for col in range(image_features.shape[1]):
    mi = mutual_information_2d(image_features[:,col].squeeze().cpu().numpy(), gender)
    mis.append((mi, col))
mis = sorted(mis, reverse=False)
mis = np.array([l[1] for l in mis])

num_clip = 400
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features  /= text_features.norm(dim=-1, keepdim=True)
image_features = image_features.cpu().numpy()[:, mis[:num_clip]]
text_features = text_features.cpu().numpy()[:, mis[:num_clip]]
sim = text_features @ image_features.T

recall, bias = evaluate(sim)
print("Recall: ", recall)
print("Bias: ", bias)
print(f"Time cost: {time.time()-begin}")
