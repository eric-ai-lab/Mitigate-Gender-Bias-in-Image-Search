import numpy as np
import pandas as pd
import torch
import clip
from PIL import Image
import requests
from utils import mutual_information_2d

device = "cuda"
model, transform = clip.load("ViT-B/32", device=device)

df = pd.read_csv('data/selected_images.csv')
occupations = df.search_term.unique()

for occupation in occupations:
    image_urls = df.image_url[df['search_term'] == occupation]
    images = torch.stack([transform(Image.open(requests.get(url, stream=True).raw)).to(device) for url in image_urls])
    # get the gender labels
    A = np.where(df.image_gender[df['search_term'] == occupation].values == 'man', 1, -1)

    with torch.no_grad():
        image_features = model.encode_image(images).float().cpu().numpy()

    # estimate mutual information
    mis = []
    for col in range(image_features.shape[1]):
        mi = mutual_information_2d(image_features[:,col].squeeze(), A)
        mis.append((mi, col))
    mis = sorted(mis, reverse=False)
    mis = np.array([l[1] for l in mis])
    
    male_image_urls = df[(df['search_term'] == occupation) & (df['image_gender'] == 'man')].image_url
    female_image_urls = df[(df['search_term'] == occupation) & (df['image_gender'] == 'woman')].image_url
    male_image = torch.stack([transform(Image.open(requests.get(url, stream=True).raw)).to(device) for url in male_image_urls])
    female_image = torch.stack([transform(Image.open(requests.get(url, stream=True).raw)).to(device) for url in female_image_urls])
    text = clip.tokenize(occupation).to(device)

    with torch.no_grad():
        male_image_features = model.encode_image(male_image).float()
        female_image_features = model.encode_image(female_image).float()
        text_features = model.encode_text(text).float()
    
    male_image_features = male_image_features.cpu().numpy()[:, mis[:400]]
    female_image_features = female_image_features.cpu().numpy()[:, mis[:400]]
    text_features = text_features.cpu().numpy()[:, mis[:400]]

    sim_male = text_features @ male_image_features.T
    sim_female = text_features @ female_image_features.T

    print(f"{occupation}\t{sim_female.mean() - sim_male.mean():.6f}")
