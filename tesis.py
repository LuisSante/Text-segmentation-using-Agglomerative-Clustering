import shutil as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram
#import scipy.cluster.hierarchy as sch

ds = pd.read_csv('https://raw.githubusercontent.com/LuisSante/Datasets/main/app_reviews.csv')

model = SentenceTransformer('all-MiniLM-L6-v2')

list_ = []  
for i in range(len(ds['package_name'].unique())):
    ds_aux = ds.loc[ds['package_name'] == ds['package_name'].unique()[i]]
    list_.append({'package_name':ds['package_name'].unique()[i], 'size': len(ds_aux)})

list_ = sorted(list_, key=lambda x: x['size'], reverse=True)
#list_experiment = list_[8]

df = ds[ds['package_name'] == list_[8]['package_name']]
sentences = list(df['review'])

embeddings = model.encode(sentences, convert_to_tensor=True)
cosine_scores = util.cos_sim(embeddings, embeddings)

range_n_clusters = range(2, 15)
valores_medios_silhouette = []

for k in range(2,15,1):
    HCA = AgglomerativeClustering(
        n_clusters = k, 
        linkage = 'complete'
        )

    HCA.fit(cosine_scores)
    #cluster_labels = HCA.fit_predict(X_scaled)
    #silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    #valores_medios_silhouette.append(silhouette_avg)

    df['Clusters'] = HCA.labels_
    df = df.sort_values('Clusters')
    df.to_csv('clusters'+str(k)+'.csv',encoding='utf-8',index=False)