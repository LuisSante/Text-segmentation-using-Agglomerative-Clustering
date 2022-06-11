'''from sentence_transformers import SentenceTransformer, util
import pandas as pd

model = SentenceTransformer('all-MiniLM-L6-v2')

# Single list of sentences
df = pd.read_csv('https://raw.githubusercontent.com/LuisSante/Datasets/main/app_reviews.csv')

sentences = df['review']

#Compute embeddings
embeddings = model.encode(sentences, convert_to_tensor=True)

#Compute cosine-similarities for each sentence with each other sentence
cosine_scores = util.cos_sim(embeddings, embeddings)

#Find the pairs with the highest cosine similarity scores
pairs = []
for i in range(len(cosine_scores)-1):
    for j in range(i+1, len(cosine_scores)):
        pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})

#Sort scores in decreasing order
pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)

for pair in pairs[0:10]:
    i, j = pair['index']
    print("{} \t\t {} \t\t Score: {:.4f}".format(sentences[i], sentences[j], pair['score']))'''


import time
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import sys
inicio = time.time()

file_path = "salida3.txt"
sys.stdout = open(file_path, 'w')

df = pd.read_csv('https://raw.githubusercontent.com/LuisSante/Datasets/main/app_reviews.csv')
model = SentenceTransformer('all-MiniLM-L6-v2')

# Single list of sentences
lista = df['review']
sentences = lista[0:10000]
paraphrases = util.paraphrase_mining(model, sentences, corpus_chunk_size=10000, top_k=100)
print(len(paraphrases))

for paraphrase in paraphrases[0:10000]:
    score, i, j = paraphrase
    print("{} \t\t {} \t\t Score: {:.4f}".format(sentences[i], sentences[j], score))

fin = time.time()
print(fin-inicio)