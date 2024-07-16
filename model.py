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
    if config['positional'] is False:
        encoded = torch.zeros((config['max_len'], config['embedding'][1]))
        if config['embedding'][0] == 'onehot':
            for i, c in enumerate(sequence):
                encoded[i][PARAPRED_TO_POS.get(c, NUM_AMINOS)] = 1
        elif config['embedding'][0] == 'berty':
            encoded[:len(sequence)] = embedding_dict[sequence].cpu()
        elif config['embedding'][0] == 'ab':
            encoded[:len(sequence)] = torch.tensor(ab_embedding_dict[sequence])
        elif config['embedding'][0] == 'prot':
            encoded[:len(sequence)] = prot_embedding_dict[sequence]
        elif config['embedding'][0] == 'balm':
            encoded[:len(sequence)] = balm_embedding_dict[sequence]
        elif config['embedding'][0] == 'ig':
            encoded[:len(sequence)] = ig_embedding_dict[sequence][:, 0, :].squeeze(1)
        elif config['embedding'][0] == 'esm':
            encoded[:len(sequence)] = esm_embedding_dict[sequence]
        else:
            print("Not Proper embedding name selected!")
            exit()

        return encoded
    else:
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


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionModule, self).__init__()

        self.branch1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.Conv1d(out_channels, out_channels, kernel_size=7, padding=3)
        )

    def forward(self, x):
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        branch3_out = self.branch3(x)
        branch4_out = self.branch4(x)

        outputs = [branch1_out, branch2_out, branch3_out, branch4_out]
        return torch.cat(outputs, dim=1)


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config

        self.inception = InceptionModule(config['embedding'][1] + 7, config['channel_size']) if config['positional'] else InceptionModule(config['embedding'][1], config['channel_size'])
        self.inception_fc = nn.Linear(config['channel_size'] * 4, 1)

        self.conv1 = nn.Conv1d(config['embedding'][1] + 7, config['channel_size'], kernel_size=7, padding=3) if config['positional'] else nn.Conv1d(config['embedding'][1], config['channel_size'], kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(config['channel_size'], config['channel_size'] // 2, kernel_size=7, padding=3)
        self.cnn_fc = nn.Linear(config['channel_size'] // 2, 1)

        self.mask_fnn1 = nn.Linear(config['embedding'][1] + 7, config['embedding'][1] // 2) if config['positional'] else nn.Linear(config['embedding'][1], config['embedding'][1] // 2)
        self.mask_fnn2 = nn.Linear(config['embedding'][1] // 2, 1)

        self.ReLU = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        # "ablation": "MASK-METHOD-FNN",  # ["MASK-FNN", "MASK-POS-FNN", "MASK-METHOD-FNN" , "MASK-POS-METHOD-FNN"]
        if self.config['ablation'] == 'MASK-FNN' or self.config['ablation'] == 'MASK-POS-FNN':
            x = self.mask_fnn1(input_tensor)
            x = self.ReLU(x)
            x = self.mask_fnn2(x)
            return self.sigmoid(x)
        elif self.config['ablation'] == 'MASK-METHOD-FNN' or self.config['ablation'] == 'MASK-POS-METHOD-FNN':
            x = input_tensor.permute([0, 2, 1])
            if self.config['method'] == 'Inception':
                x = self.inception(x)
                x = self.ReLU(x)
                x = x.permute([0, 2, 1])
                return self.sigmoid(self.inception_fc(x))
            elif self.config['method'] == 'CNN':
                x = self.conv1(x)
                x = self.ReLU(x)
                x = self.conv2(x)
                x = self.ReLU(x)
                x = x.permute([0, 2, 1])
                return self.sigmoid(self.cnn_fc(x))
