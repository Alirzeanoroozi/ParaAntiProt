from torch.utils.data import Dataset, DataLoader


class ABDataset(Dataset):
    def __init__(self, chains, labels, positions):
        self.chains = chains
        self.labels = labels
        self.pos = positions

    def __len__(self):
        return len(self.chains)

    def __getitem__(self, idx):
        return self.chains[idx], self.labels[idx], self.pos[idx]


def ab_loader(train, valid, test, config):
    return DataLoader(train, batch_size=config['batch_size']),\
        DataLoader(valid, batch_size=config['batch_size']),\
        DataLoader(test, batch_size=config['batch_size'])
