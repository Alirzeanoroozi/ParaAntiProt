import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from data.dataset import ABDataset, ab_loader


def check_all_zero(input_strs):
    return [sum([1 if i == '1' else 0 for i in input_str]) != 0 and
            sum([1 if i == '1' else 0 for i in input_str]) != len(input_str) for input_str in input_strs]


def to_binary(input_list, max_len):
    return np.array([0. if c == '0' else 1. for c in input_list] + [0. for _ in range(max_len - len(input_list))])


def to_binary_N_P(input_list, max_len):
    # print(input_list)
    input_list = input_list.replace("'", "").replace("\n", "").strip('[]').split(" ")
    # print(input_list)
    return np.array([0. if c == 'N' else 1. for c in input_list] + [0. for _ in range(max_len - len(input_list))])


def pad_position_list(position, max_len):
    return np.vstack((position, np.array([[0, 0, 0, 0, 0, 0, 0] for _ in range(max_len - len(position))])))


def zero_position_list(max_len):
    return np.array([[0, 0, 0, 0, 0, 0, 0] for _ in range(max_len)])



def get_dataloaders(test_df, train_df, val_df, config):
    chains_train = [x for x in train_df['sequence'].tolist()]
    positions_train = [pad_position_list(p, config['max_len']) for p in get_positions(train_df)]
    labels_train = [to_binary(x, config['max_len']) for x in train_df['paratope'].tolist()]

    chains_valid = [x for x in val_df['sequence'].tolist()]
    positions_valid = [pad_position_list(p, config['max_len']) for p in get_positions(val_df)]
    labels_valid = [to_binary(x, config['max_len']) for x in val_df['paratope'].tolist()]

    chains_test = [x for x in test_df['sequence'].tolist()]
    positions_test = [pad_position_list(p, config['max_len']) for p in get_positions(test_df)]
    labels_test = [to_binary(x, config['max_len']) for x in test_df['paratope'].tolist()]

    train_data = ABDataset(chains_train, labels_train, positions_train)
    valid_data = ABDataset(chains_valid, labels_valid, positions_valid)
    test_data = ABDataset(chains_test, labels_test, positions_test)

    return ab_loader(train_data, valid_data, test_data, config)


def get_positions(train_df):
    exploded = train_df['cdr_type'].apply(
        lambda s: s.replace(" '", "").replace("'", "").strip('[]').split(",")).explode()
    all_dummies_list = pd.get_dummies(exploded, dtype=float)
    if "XX" not in all_dummies_list.columns:
        all_dummies_list["XX"] = [0. for _ in range(len(all_dummies_list))]
    result = all_dummies_list.groupby(all_dummies_list.index).apply(lambda x: x.reset_index(drop=True).values).values
    return result


def get_cv_dataloaders(config, cross_round):
    # df = pd.read_csv("data/cdrs_paragraph.csv") if config['input_type'] == "cdr" else pd.read_csv("data/chains_paragraph.csv")
    df = pd.read_csv("data/cdrs_parapred.csv") if config['input_type'] == "cdr" else pd.read_csv("data/chains_parapred.csv")

    df = df[check_all_zero(df['paratope'])]
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    train_index, test_index = list(kf.split(range(len(df))))[cross_round]
    train_df, test_df = df.iloc[train_index], df.iloc[test_index]
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

    return get_dataloaders(test_df, train_df, val_df, config)
