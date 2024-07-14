# ParaAntiProt-Paratope-Prediction-Using-Antibody-and-Protein-Language-Models

The original implementation of methodology described in "ParaAntiProt: Paratope Prediction Using Antibody and Protein Language Models".

We would also like to point users to the original Github repo for the implementation of Parapred, based on Pytorch that we used their code to build this repo.

## Install

Requirements:
   * Python 3.11+ 

To install:
   * Run `python setup.py install` in the root directory. If you are using a Python installation
     manager, such as Anaconda or Canopy, follow their package installation instructions.
   * If you do not wish to install and run Parapred directly from a clone of this repository instead,
     install required packages using `pip install -r requirements.txt`.

## Usage
   * you can use the notebook for the prediction and training model and more stuff.
   * If you choose to run ParaAntiProt directly, make sure you've installed required packages from
     `requirements.txt`.

```
ParaAntiProt-Paratope-Prediction-Using-Antibody-and-Protein-Language-Models

ParaAntiProt works on antibody's amino acid sequence. The program will output
binding probability for every residue in the input. The program accepts two
kinds of input (see usage section below for examples):

(a) The full sequence of a VH or VL domain, or a larger stretch of the sequence
    of either the heavy or light chain comprising the CDR loops.

(b) An amino acid sequence corresponding to a CDR with 2 extra residues on
    either side. and specifying which part does it blongs to, e.g. (ARSGYYGDSDWYFDVGG, L1)

    Multiple CDR sequences can be processed at once by specifying a file,
    containing each sequence and CDR positions on a separate line.
```

## Training
change the config dictionary in the `main.py` and run it for trainning the results of cv will be at results folder

```
config = {
    "embedding": ("berty", 512), # [("berty", 512), ("ab", 768), ("prot", 1024), ("onehot", 21), ("balm", 640), ("esm", 1280), ("ig", 1024)]
    "input_type": "cdr",  # ["cdr", "chain"]
    "max_len": 35,  # [35, 150]

    "dataset": "parapred",  # ['nano', 'paragraph', 'parapred']
    "method": "CNN",  # ['CNN', 'LSTM', 'Inception']
    "positional": True,
    "ablation": "MASK-POS-METHOD-FNN",  # ["MASK-FNN", "MASK-POS-FNN", "MASK-METHOD-FNN" , "MASK-POS-METHOD-FNN"]

    "channel_size": 512,

    "batch_size": 16,
    "lr": 0.001,
    "num_epochs": 30
}
```

