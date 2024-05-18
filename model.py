import torch.nn as nn
import pickle
import torch


PARAPRED_AMINO = "CSTPAGNDEQHRKMILVFYW-"
PARAPRED_TO_POS = dict([(v, i) for i, v in enumerate(PARAPRED_AMINO)])
NUM_AMINOS = len(PARAPRED_AMINO)

with open("embeddings/embeddings.p", "rb") as f:
    embedding_dict = pickle.load(f)

with open("embeddings/ab_embeddings.p", "rb") as f:
    ab_embedding_dict = pickle.load(f)

with open("embeddings/prot_embeddings.p", "rb") as f:
    prot_embedding_dict = pickle.load(f)

with open("embeddings/balm_embeddings.p", "rb") as f:
    balm_embedding_dict = pickle.load(f)

with open("embeddings/esm_embeddings.p", "rb") as f:
    esm_embedding_dict = pickle.load(f)

with open("embeddings/ig_embeddings.p", "rb") as f:
    ig_embedding_dict = pickle.load(f)


def encode(sequence, pos, config):
    encoded = torch.zeros((config['max_len'], config['embedding'][1] + 7))
    if config['embedding'][0] == 'onehot':
        for i, c in enumerate(sequence):
            encoded[i][PARAPRED_TO_POS.get(c, NUM_AMINOS)] = 1
    elif config['embedding'][0] == 'berty':
        encoded[:len(sequence)] = torch.cat((embedding_dict[sequence].cpu(), pos[:len(sequence)]), dim=1)
    elif config['embedding'][0] == 'ab':
        encoded[:len(sequence)] = torch.cat((torch.tensor(ab_embedding_dict[sequence]), pos[:len(sequence)]), dim=1)
    elif config['embedding'][0] == 'prot':
        encoded[:len(sequence)] = torch.cat((prot_embedding_dict[sequence], pos[:len(sequence)]), dim=1)
    elif config['embedding'][0] == 'balm':
        encoded[:len(sequence)] = torch.cat((balm_embedding_dict[sequence], pos[:len(sequence)]), dim=1)
    elif config['embedding'][0] == 'ig':
        encoded[:len(sequence)] = torch.cat((ig_embedding_dict[sequence][:, 0, :].squeeze(1), pos[:len(sequence)]), dim=1)
    elif config['embedding'][0] == 'esm':
        encoded[:len(sequence)] = torch.cat((esm_embedding_dict[sequence], pos[:len(sequence)]), dim=1)
    else:
        print("Not Proper embedding name selected!")
        exit()

    return encoded


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.conv1 = nn.Conv1d(config['embedding'][1] + 7, config['channel_size'], kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(config['channel_size'], config['channel_size'] // 2, kernel_size=7, padding=3)
        self.ReLU = nn.functional.relu
        self.cnn_fc = nn.Linear(config['channel_size'] // 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        x = self.conv1(input_tensor.permute([0, 2, 1]))
        x = self.ReLU(x)
        x = self.conv2(x)
        x = self.ReLU(x)
        x = x.permute([0, 2, 1])
        return self.sigmoid(self.cnn_fc(x))
