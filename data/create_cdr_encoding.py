import pandas as pd

from utils import split_consecutive_elements


df = pd.read_csv("processed_dataset_test.csv")

cdr_annotations = []

for s, chain_type, cdr in zip(df['sequence'], df['chain_type'], df['cdrs']):
    annotation = []
    cdr_indexes = split_consecutive_elements([int(c) for c in cdr.strip("][").split(",")])
    for i, c in enumerate(s):
        x = 'XX'
        for j in range(3):
            if i in cdr_indexes[j]:
                x = chain_type + str(j + 1)
        annotation.append(x)
    cdr_annotations.append(annotation)

df['cdr_type'] = cdr_annotations
df.to_csv("chains_test.csv", index=False)
