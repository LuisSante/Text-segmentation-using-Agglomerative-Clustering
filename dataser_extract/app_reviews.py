from datasets import list_datasets, load_dataset
from pprint import pprint
import pandas as pd

app_reviews = load_dataset('app_reviews')
print(app_reviews)

app_reviews = load_dataset('app_reviews', split='train')

print("Length of training set: ", len(app_reviews))
print("First example from the dataset: \n")
pprint(app_reviews[0])

print("Features: ")
pprint(app_reviews.features)
print("Column names: ", app_reviews.column_names)

print("Number of rows: ", app_reviews.num_rows)
print("Number of columns: ", app_reviews.num_columns)
print("Shape: ", app_reviews.shape)

app_reviews_ds = pd.DataFrame(app_reviews)
app_reviews_ds.to_csv('app_reviews.csv',index=False)