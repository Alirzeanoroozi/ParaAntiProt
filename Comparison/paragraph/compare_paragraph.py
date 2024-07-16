import pandas as pd


# df_append = pd.DataFrame()
#
# csv_files = ['train_set.csv', 'val_set.csv', 'test_set.csv']
# for file in csv_files:
#             df_temp = pd.read_csv(file, header=None)
#             df_append = df_append._append(df_temp, ignore_index=True)
# df_append.to_csv("Paragraph.csv", index=False, header=['pdb', 'Hchain', 'Lchain', 'antigen_chain'])
# exit()

df = pd.read_csv("example_predictions.csv")
# pdb,chain_type,chain_id,IMGT,AA,atom_num,x,y,z,pred

processed_df = df[['pdb', 'chain_type', 'IMGT', 'pred']]

processed_df.to_csv("processed_predictions.csv")

grouped_pred = df.groupby(['pdb', 'chain_id'])['pred'].apply(lambda x: x.reset_index(drop=True).values)
grouped_imgt = df.groupby(['pdb', 'chain_id'])['IMGT'].apply(lambda x: x.reset_index(drop=True).values)

print(grouped_imgt )
print(grouped_pred )
