import pandas as pd

from utils import split_consecutive_elements


df = pd.read_csv("processed_dataset_test.csv")

pdbs = []
cdr_types = []
cdr_sequences = []
cdr_paratopes = []
for _, row in df.iterrows():
    pdbs.extend([row['pdb'], row['pdb'], row['pdb']])
    cdr_parts = split_consecutive_elements([int(c) for c in row['cdrs'].strip("][").split(",")])
    cdr_types.extend([
        [row['chain_type'] + '1' for _ in range(len(cdr_parts[0]))],
        [row['chain_type'] + '2' for _ in range(len(cdr_parts[1]))],
        [row['chain_type'] + '3' for _ in range(len(cdr_parts[2]))],
    ])
    for cdr in cdr_parts:
        cdr_sequences.append("".join([row['sequence'][i] for i in cdr]))
        cdr_paratopes.append("".join([row['paratope'][i] for i in cdr]))

new_df = pd.DataFrame(
    {
        'pdbs': pdbs,
        'cdr_type': cdr_types,
        'sequence': cdr_sequences,
        'paratope': cdr_paratopes
    }
).to_csv("cdrs_test.csv", index=False)
