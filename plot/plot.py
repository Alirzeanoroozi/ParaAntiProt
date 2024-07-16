import pickle
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

with open('saved_dictionary.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

d2 = {k: [torch.cat(v1, dim=0).tolist() for v1 in v] for k, v in loaded_dict.items()}

print(d2.keys())
map_name = {
    "berty": "AntiBERTy",
    "ab": "AbLang",
    "prot": "ProtTrans",
    "balm": "Balm",
    "esm": "ESM-2",
    "ig": "Ig BERT"
}


def plot_roc_curve(plot_dict):
    for model in ["berty", "ab", "prot", "balm", "esm"]:
        d2 = {k: v for k, v in plot_dict.items() if model in k}
        all_labels = []
        all_probs = []
        for k, (efective_labels, efective_probs) in d2.items():
            all_labels.extend(efective_labels)
            all_probs.extend(efective_probs)
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        plt.plot(fpr, tpr, label=f"ParaAntiProt({map_name[model]})")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc=4)
    plt.show()
    plt.savefig(f"../figures/ROC_Curve.tiff", dpi=300, format="tiff", pil_kwargs={"compression": "tiff_lzw"})


def plot_pr_curve(plot_dict):
    for model in ["berty", "ab", "prot", "balm", "esm"]:
        d2 = {k: v for k, v in plot_dict.items() if model in k}
        all_labels = []
        all_probs = []
        for k, (efective_labels, efective_probs) in d2.items():
            all_labels.extend(efective_labels)
            all_probs.extend(efective_probs)
        precision1, recall1, thresholds1 = precision_recall_curve(all_labels, all_probs)
        plt.plot(recall1, precision1, label=f"ParaAntiProt({map_name[model]})")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall (PR) Curve')
    plt.legend(loc="lower left")
    plt.show()
    plt.savefig(f"../figures/PR_Curve.tiff", dpi=300, format="tiff", pil_kwargs={"compression": "tiff_lzw"})


plot_roc_curve(d2)
plot_pr_curve(d2)


method_names = ['ab_512', 'balm_512', 'berty_512', 'esm_512', 'ig_512', 'prot_512']
metrics = ['recall', 'precision', 'f1', 'roc', 'pr', 'mcc']


def plot_metric():
    for m in metrics:
        values = []
        for n in method_names:
            a_values = []
            for i in range(10):
                with open(f"../results/cdr/parapred_MASK-POS-METHOD-FNN_CNN_{n}/cv_{i}.txt", 'r') as file:
                    lines = file.readlines()
                    a_values.append(float(lines[metrics.index(m)].split()[2]))  # Extracting the second value
            values.append(a_values)

        plt.boxplot(values, labels=['ParaAntiProt(' + f + ')' for f in method_names])
        plt.xticks(rotation=60)
        xs = [np.random.normal(i + 1, 0.04, 10) for i in range(len(values))]

        palette = [
            'r', 'g', 'b', 'y',  # Red, Green, Blue, Yellow
            'c', 'm', 'k', 'w',  # Cyan, Magenta, Black, White
            '#FF5733', '#33FF57',  # Custom Hex Colors
        ]

        for x, val, c in zip(xs, values, palette):
            plt.scatter(x, val, alpha=0.4, color=c)
        plt.title(f'{m}')
        plt.tight_layout()
        plt.savefig(f'../figures/metrics/{m}.tiff', dpi=600, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
        plt.show()


def plot_ablation():
    for method_name in ['ab_512', 'balm_512', 'berty_512', 'esm_512', 'ig_512', 'prot_512']:
        for metric in metrics:
            print(f'{method_name}_{metric}')
            all_values = []
            for a in ["MASK-FNN", "MASK-POS-FNN", "MASK-METHOD-FNN_CNN", "MASK-POS-METHOD-FNN_CNN"]:
                values = []
                for i in range(10):
                    with open(f"../results/cdr/parapred_{a}_{method_name}/cv_{i}.txt", 'r') as file:
                        lines = file.readlines()
                        values.append(float(lines[metrics.index(metric)].split()[2]))  # Extracting the second value
                all_values.append(values)

            # plt.figure(figsize=(6, 6))
            plt.plot()
            plt.boxplot(all_values, labels=["MASK-FNN", "MASK-POS-FNN", "MASK-CNN-FNN", "MASK-POS-CNN-FNN"],
                        vert=False)
            xs = [np.random.normal(i + 1, 0.04, 10) for i in range(len(all_values))]
            palette = [
                'r', 'g', 'b', 'y',  # Red, Green, Blue, Yellow
                'c', 'm', 'k', 'w',  # Cyan, Magenta, Black, White
            ]

            for x, val, c in zip(xs, all_values, palette):
                plt.scatter(val, x, alpha=0.4, color=c)

            plt.yticks(rotation=70)
            plt.title(f'{method_name}_{metric}')
            plt.savefig(f'../figures/ablation/{method_name}_{metric}.tiff', dpi=600, format="tiff",
                        pil_kwargs={"compression": "tiff_lzw"})
            plt.tight_layout()
            plt.show()

#
# plot_metric()
# plot_ablation()
