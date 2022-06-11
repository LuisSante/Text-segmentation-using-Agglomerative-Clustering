from datasets import list_datasets, load_dataset
from pprint import pprint
import pandas as pd

banca = load_dataset('banking77')
print(banca)

banca = load_dataset('banking77', split='train')

print("Length of training set: ", len(banca))
print("First example from the dataset: \n")
pprint(banca[0])

print("Features: ")
pprint(banca.features)
print("Column names: ", banca.column_names)

print("Number of rows: ", banca.num_rows)
print("Number of columns: ", banca.num_columns)
print("Shape: ", banca.shape)

banca_ds = pd.DataFrame(banca)
banca_ds.to_csv('banca.csv',index=False)