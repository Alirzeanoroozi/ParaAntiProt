import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.data_loader import pad_position_list, get_positions, to_binary
from data.dataset import ABDataset
from model import encode, Model
from train_file import evaluate
from utils import report_cv


def test_paragraph(config, device, method_name):
    test_df = pd.read_csv("data/cdrs_test.csv")
    chains_test = [x for x in test_df['sequence'].tolist()]
    positions_test = [pad_position_list(p, 35) for p in get_positions(test_df)]
    labels_test = [to_binary(x, 35) for x in test_df['paratope'].tolist()]
    test_data = ABDataset(chains_test, labels_test, positions_test)
    test_dataloader = DataLoader(test_data, batch_size=config['batch_size'])

    para_method_name = "paragraph" + method_name

    if not os.path.exists(f"results/cdr/{para_method_name}"):
        os.mkdir(f"results/cdr/{para_method_name}")

    for i in range(10):
        # model = Model(config).to(device)
        model = torch.load(f"trained_models/cdr/{method_name}/{i}.pth")

        with open(f"trained_models/cdr/{method_name}/{i}_tresh.txt", "r") as f:
            tresh = float(f.read())

        _, eval_list = evaluate(model, encode, test_dataloader, device, config, "test", para_method_name, tresh, i)

    report_cv(f"results/cdr/{para_method_name}/", "cv")


# AB
# Roc  0.845, pr   0.678

# BALM
# Roc  0.893, pr   0.721

# BERTy
# Roc  0.908, pr   0.739
