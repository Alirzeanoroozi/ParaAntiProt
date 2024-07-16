import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from data.data_loader import to_binary_N_P
from data.dataset import ABDataset
from model import encode, Model
from train_file import evaluate, train
from utils import initiate_system_device, save_models

device = initiate_system_device()
config = {
    "embedding": ("balm", 640),
    # [("berty", 512), ("ab", 768), ("prot", 1024), ("onehot", 21), ("balm", 640), ("esm", 1280), ("ig", 1024)]
    "input_type": "chain",  # ["cdr", "chain"]
    "max_len": 150,  # [35, 150]

    "channel_size": 512,

    "batch_size": 16,
    "lr": 0.001,
    "num_epochs": 30
}

train_df = pd.read_csv("data/NanoBERTa/nanotrain.csv")
val_df = pd.read_csv("data/NanoBERTa/nanoval.csv")
test_df = pd.read_csv("data/NanoBERTa/nanotest.csv")

chains_train = [x for x in train_df['sequence'].tolist()]
positions_train = [np.array([[0, 0, 0, 0, 0, 0, 0] for _ in range(config['max_len'])]) for _ in train_df['sequence'].tolist()]
labels_train = [to_binary_N_P(x, config['max_len']) for x in train_df['paratope_labels'].tolist()]

chains_valid = [x for x in val_df['sequence'].tolist()]
positions_valid = [np.array([[0, 0, 0, 0, 0, 0, 0] for _ in range(config['max_len'])]) for _ in val_df['sequence'].tolist()]
labels_valid = [to_binary_N_P(x, config['max_len']) for x in val_df['paratope_labels'].tolist()]

chains_test = [x for x in test_df['sequence'].tolist()]
positions_test = [np.array([[0, 0, 0, 0, 0, 0, 0] for _ in range(config['max_len'])]) for _ in test_df['sequence'].tolist()]
labels_test = [to_binary_N_P(x, config['max_len']) for x in test_df['paratope_labels'].tolist()]

train_data = ABDataset(chains_train, labels_train, positions_train)
valid_data = ABDataset(chains_valid, labels_valid, positions_valid)
test_data = ABDataset(chains_test, labels_test, positions_test)

train_dataloader, valid_dataloader, test_dataloader = DataLoader(train_data, batch_size=config['batch_size']), DataLoader(valid_data, batch_size=config['batch_size']), DataLoader(test_data, batch_size=config['batch_size'])
# exit()
nano_method_name = "NanoBERTa_" + config['embedding'][0] + "_" + str(config['channel_size'])

if not os.path.exists(f"results/chain/{nano_method_name}"):
    os.mkdir(f"results/chain/{nano_method_name}")
if not os.path.exists(f"trained_models/{config['input_type']}/{nano_method_name}"):
    os.mkdir(f"trained_models/{config['input_type']}/{nano_method_name}")

model = Model(config).to(device)

model, tresh, losses = train(model, encode, train_dataloader, valid_dataloader, device, config, nano_method_name, 0)
save_models(model, tresh, config, nano_method_name, 0)

_, eval_list = evaluate(model, encode, test_dataloader, device, config, "test", nano_method_name, tresh, 0)
