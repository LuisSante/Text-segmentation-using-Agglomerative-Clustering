import shutil as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram, linkage , fcluster
from matplotlib import pyplot as plt

ds = pd.read_csv('https://raw.githubusercontent.com/LuisSante/Datasets/main/app_reviews.csv')

model = SentenceTransformer('all-MiniLM-L6-v2')

list_ = []  
for i in range(len(ds['package_name'].unique())):
    ds_aux = ds.loc[ds['package_name'] == ds['package_name'].unique()[i]]
    list_.append({'package_name':ds['package_name'].unique()[i], 'size': len(ds_aux)})

list_ = sorted(list_, key=lambda x: x['size'], reverse=True)

while(len(list_) > 10):
    list_.pop(len(list_)-6)

for i in range(5,7,1):
    df = []
    df = ds[ds['package_name'] == list_[i]['package_name']]
    sentences = list(df['review'])

    embeddings = model.encode(sentences, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings, embeddings)

    HCA = linkage(cosine_scores, 'complete')

    plt.figure(figsize=(10,10))
    dendrograma = sch.dendrogram(HCA)
    plt.savefig('clusters' + str(i) + '.png')

    clusters = fcluster(HCA, t=2, criterion='distance')

    df['Clusters'] = clusters
    df.to_csv('clusters'+str(i)+'.csv',encoding='utf-8',index=False)