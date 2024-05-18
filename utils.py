import random
import numpy as np
import torch


def annotate_cdrs(list1):
    result = []
    current_group = []
    for item in list1:
        if not current_group or item == current_group[-1] + 1:
            current_group.append(item)
        else:
            result.append(current_group)
            current_group = [item]
    result.append(current_group)  # Add the last group
    return result


def save_models(model, tresh, config, method_name, cv):
    torch.save(model, f"trained_models/{config['input_type']}/{method_name}/{cv}.pth")
    with open(f"trained_models/{config['input_type']}/{method_name}/{cv}_tresh.txt", "w") as f:
        f.write(str(tresh))


def split_consecutive_elements(list1):
    result = []
    current_group = []
    for item in list1:
        if not current_group or item == current_group[-1] + 1:
            current_group.append(item)
        else:
            result.append(current_group)
            current_group = [item]
    result.append(current_group)  # Add the last group
    return result


def report_cv(file_path, file_name):
    recall_values = []
    precision_values = []
    f1_values = []
    roc_values = []
    pr_values = []
    mcc_values = []

    for i in range(10):
        file_path_ = f"{file_path}{file_name}_{i}.txt"

        with open(file_path_, 'r') as file:
            lines = file.readlines()

            recall_line = lines[0]  # Assuming Precision is in the 7th line
            recall_values.append(float(recall_line.split()[2]))  # Extracting the second value

            precision_line = lines[1]  # Assuming Precision is in the 7th line
            precision_values.append(float(precision_line.split()[2]))  # Extracting the second value

            f1_line = lines[2]  # Assuming Precision is in the 7th line
            f1_values.append(float(f1_line.split()[2]))  # Extracting the second value

            roc_line = lines[3]  # Assuming Precision is in the 7th line
            roc_values.append(float(roc_line.split()[2]))  # Extracting the second value

            pr_line = lines[4]  # Assuming Precision is in the 7th line
            pr_values.append(float(pr_line.split()[2]))  # Extracting the second value

            mcc_line = lines[5]  # Assuming Precision is in the 7th line
            mcc_values.append(float(mcc_line.split()[2]))  # Extracting the second value

    with open(f"{file_path}{file_name}.txt", "w") as f:
        f.writelines([
            "recall: " + str(round(np.nanmean(recall_values), 3)) + " +/- " + str(round(np.nanstd(recall_values), 3)) + "\n",
            "precision: " + str(round(np.nanmean(precision_values), 3)) + " +/- " + str(round(np.nanstd(precision_values), 3)) + "\n",
            "F1-Score: " + str(round(np.nanmean(f1_values), 3)) + " +/- " + str(round(np.nanstd(f1_values), 3)) + "\n",
            "ROC: " + str(round(np.nanmean(roc_values), 3)) + " +/- " + str(round(np.nanstd(roc_values), 3)) + "\n",
            "PR-score: " + str(round(np.nanmean(pr_values), 3)) + " +/- " + str(round(np.nanstd(pr_values), 3)) + "\n",
            "MCC: " + str(round(np.nanmean(mcc_values), 3)) + " +/- " + str(round(np.nanstd(mcc_values), 3)) + "\n",
        ])


def initiate_system_device():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    torch.backends.cudnn.benchmarks = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def encode_batch(batch_of_sequences, positions, encoder, config):
    embeddings = [encoder(seq, pos, config) for seq, pos in zip(batch_of_sequences, positions)]
    seq_lens = [len(seq) for seq in batch_of_sequences]
    return torch.stack(embeddings), torch.as_tensor(seq_lens)

def check_all_zero(input_strs):
    return [sum([1 if i == '1' else 0 for i in input_str]) >= 1 for input_str in input_strs]

