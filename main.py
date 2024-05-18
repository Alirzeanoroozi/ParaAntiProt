import os
import warnings

import torch

from Comparison.paragraph.test_paragraph import test_paragraph
from data.data_loader import get_cv_dataloaders
from model import Model, encode
from train_file import train, evaluate
from utils import initiate_system_device, report_cv, save_models

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    device = initiate_system_device()
    config = {
        "embedding": ("onehot", 21),
        # [("berty", 512), ("ab", 768), ("prot", 1024), ("onehot", 21), ("balm", 640), ("esm", 1280), ("ig", 1024)]
        "input_type": "cdr",  # ["cdr", "chain"]
        "max_len": 35,  # [35, 150]

        "channel_size": 512,

        "batch_size": 16,
        "lr": 0.001,
        "num_epochs": 30
    }
    method_name = config['embedding'][0] + "_" + str(config['channel_size'])

    # test_paragraph(config, device, method_name)
    # exit()

    plot_dict = {}

    for e in [("berty", 512), ("ab", 768), ("prot", 1024), ("balm", 640), ("esm", 1280), ("ig", 1024)]:
        config['embedding'] = e

        print(e)

        method_name = config['embedding'][0] + "_" + str(config['channel_size'])

        if not os.path.exists(f"results/{config['input_type']}/{method_name}"):
            os.mkdir(f"results/{config['input_type']}/{method_name}")
        if not os.path.exists(f"trained_models/{config['input_type']}/{method_name}"):
            os.mkdir(f"trained_models/{config['input_type']}/{method_name}")

        for i in range(10):
            print("Cross_validation run", i + 1)
            train_dataloader, test_dataloader, valid_dataloader = get_cv_dataloaders(config, i)
            # model = Model(config).to(device)
            #
            # model, tresh, losses = train(model, encode, train_dataloader, valid_dataloader, device, config, method_name, i)
            # save_models(model, tresh, config, method_name, i)
            model = torch.load(f"trained_models/cdr/{method_name}/{i}.pth")

            with open(f"trained_models/cdr/{method_name}/{i}_tresh.txt", "r") as f:
                tresh = float(f.read())
            _, eval_list = evaluate(model, encode, test_dataloader, device, config, "test", method_name, tresh, i)
            plot_dict[method_name + str(i + 1)] = [eval_list['eff_labels'], eval_list['eff_probs']]

        report_cv(f"results/{config['input_type']}/{method_name}/", "cv")
        import pickle

        with open('plot/saved_dictionary.pkl', 'wb') as f:
            pickle.dump(plot_dict, f)



