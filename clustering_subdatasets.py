import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram, linkage , fcluster
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

ds = pd.read_csv('https://raw.githubusercontent.com/LuisSante/Datasets/main/app_reviews.csv')

model = SentenceTransformer('all-MiniLM-L6-v2')

current = ds['package_name'] == ds['package_name'].unique()[0]
ds_aux = ds[current]

corpus = ds_aux['review']
sentences = corpus[0:len(ds_aux)]

embeddings = model.encode(sentences, convert_to_tensor=True)
cosine_scores = util.cos_sim(embeddings, embeddings)

HCA = linkage(cosine_scores, 'complete')
plt.figure(figsize=(100,100))
dendrograma = sch.dendrogram(HCA)
plt.savefig('clusters.png')

clusters = fcluster(HCA, t=2.5, criterion='distance')

ds_aux['Clusters'] = clusters
ds_clus = ds_aux.sort_values('Clusters')
ds_clus.to_csv('clusters.csv',encoding='utf-8',index=False)