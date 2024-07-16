import random
import numpy as np
import pandas as pd
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


def load_models(config, i, method_name):
    model = torch.load(f"trained_models/{config['input_type']}/{method_name}/{i}.pth")
    with open(f"trained_models/{config['input_type']}/{method_name}/{i}_tresh.txt", "r") as f:
        tresh = np.float64(f.read())
    return model, tresh


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


def create_df_results(input_arg):
    df = pd.DataFrame()
    if input_arg == 'cdr':
        for a in ["MASK-FNN", "MASK-POS-FNN", "MASK-METHOD-FNN_CNN", "MASK-POS-METHOD-FNN_CNN",
                  "MASK-POS-METHOD-FNN_Inception"]:
            for method_name in ['ab_512', 'balm_512', 'berty_512', 'esm_512', 'ig_512', 'prot_512']:
                if a == "MASK-METHOD-FNN_CNN" and method_name == 'ig_512':
                    continue
                with open(f"results/cdr/parapred_{a}_{method_name}/cv.txt", 'r') as file:
                    lines = file.readlines()
                    df = df._append({
                        "method_name": f'{a}_{method_name}',
                        'recall': lines[0].split(": ")[1].strip(),
                        'precision': lines[1].split(": ")[1].strip(),
                        'f1': lines[2].split(": ")[1].strip(),
                        'roc': lines[3].split(": ")[1].strip(),
                        'pr': lines[4].split(": ")[1].strip(),
                        'mcc': lines[5].split(": ")[1].strip(),
                    }, ignore_index=True)
        df.to_csv("cdr_results.csv", index=False)
    else:
        for a in ["MASK-POS-METHOD-FNN_CNN", "MASK-POS-METHOD-FNN_Inception"]:
            for method_name in ['ab_512', 'balm_512', 'berty_512', 'onehot_512']:
                with open(f"results/chain/parapred_{a}_{method_name}/cv.txt", 'r') as file:
                    lines = file.readlines()
                    df = df._append({
                        "method_name": f'{a}_{method_name}',
                        'recall': lines[0].split(": ")[1].strip(),
                        'precision': lines[1].split(": ")[1].strip(),
                        'f1': lines[2].split(": ")[1].strip(),
                        'roc': lines[3].split(": ")[1].strip(),
                        'pr': lines[4].split(": ")[1].strip(),
                        'mcc': lines[5].split(": ")[1].strip(),
                    }, ignore_index=True)
        df.to_csv("chain_results.csv", index=False)


def create_cdr_encoding():
    df = pd.read_csv("data/processed_dataset_test.csv")

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
    df.to_csv("data/chains_test.csv", index=False)


def convert_chains_cdr():
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

    pd.DataFrame(
        {
            'pdbs': pdbs,
            'cdr_type': cdr_types,
            'sequence': cdr_sequences,
            'paratope': cdr_paratopes
        }
    ).to_csv("cdrs_test.csv", index=False)


def select_best_models(input_arg):
    if input_arg == 'cdr':
        for n in ['ab_512', 'balm_512', 'berty_512', 'esm_512', 'ig_512', 'prot_512']:
            a_values = []
            for i in range(10):
                with open(f"results/cdr/parapred_MASK-POS-METHOD-FNN_CNN_{n}/cv_{i}.txt", 'r') as file:
                    lines = file.readlines()
                    a_values.append(float(lines[4].split()[2]))
            i = a_values.index(max(a_values))
            model = torch.load(f"trained_models/cdr/parapred_MASK-POS-METHOD-FNN_CNN_{n}/{i}.pth")
            with open(f"trained_models/cdr/parapred_MASK-POS-METHOD-FNN_CNN_{n}/{i}_tresh.txt", "r") as f:
                tresh = f.read()

            name = n.split("_")[0]
            torch.save(model, f"best_models/cdr_{name}.pth")
            with open(f"best_models/cdr_{name}_tresh.txt", "w") as f:
                f.write(str(tresh))
    else:
        for n in ['ab_512', 'balm_512', 'berty_512']:
            a_values = []
            for i in range(10):
                with open(f"results/chain/parapred_MASK-POS-METHOD-FNN_CNN_{n}/cv_{i}.txt", 'r') as file:
                    lines = file.readlines()
                    a_values.append(float(lines[4].split()[2]))
            i = a_values.index(max(a_values))
            model = torch.load(f"trained_models/chain/parapred_MASK-POS-METHOD-FNN_CNN_{n}/{i}.pth")
            with open(f"trained_models/chain/parapred_MASK-POS-METHOD-FNN_CNN_{n}/{i}_tresh.txt", "r") as f:
                tresh = f.read()

            name = n.split("_")[0]
            torch.save(model, f"best_models/chain_{name}.pth")
            with open(f"best_models/chain_{name}_tresh.txt", "w") as f:
                f.write(str(tresh))
