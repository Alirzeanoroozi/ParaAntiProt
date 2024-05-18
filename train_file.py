import torch
import torch.nn as nn
from torch import optim

from evaluation import compute_classifier_metrics
from utils import encode_batch


def train(model, encoder, train_dl, val_dl, device, config, file_name, cv):
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    loss_fn = nn.BCELoss()
    losses = {"train": [], "val": []}

    # Initialize early stopping parameters
    patience = 5
    early_stopping_counter = 0

    best_validation_loss = float('inf')
    best_model = None
    best_tresh = 0

    # Initialize learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    for epoch in range(config['num_epochs']):
        model.train()
        train_loss = 0.0

        for batch in train_dl:
            chains, labels, poses = batch

            optimizer.zero_grad()

            embeddings, lengths = encode_batch(chains, poses, encoder, config)
            sequences = embeddings.to(device)
            probabilities = model(sequences)
            out = probabilities.squeeze(2).type(torch.float64).cpu()

            loss = loss_fn(out, labels)
            train_loss += loss.data.item()
            loss.backward()

            optimizer.step()

        train_loss = train_loss / len(train_dl)
        val_loss, eval = evaluate(model, encoder, val_dl, device, config, epoch, file_name, cv=cv)

        # Check if the validation loss has improved
        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
            best_model = model
            best_tresh = eval['Youden']

            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        # Adjust learning rate with the scheduler
        scheduler.step(val_loss)

        # Check if we should early stop
        if early_stopping_counter >= patience:
            print("Early stopping triggered.")
            break

        print('Epoch %3d/%3d, train loss: %5.2f, val loss: %5.2f' % (epoch + 1, config['num_epochs'], train_loss, val_loss))
        losses["train"].append(train_loss)
        losses["val"].append(val_loss)

    return best_model, best_tresh, losses


def evaluate(model, encoder, loader, device, config, epoch=None, file_name=None, threshold=None, cv=None):
    loss_fn = nn.BCELoss()
    model.eval()
    model = model.to(device)

    val_loss = 0.0
    all_outs = []
    all_lengths = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            chains, labels, poses = batch

            embeddings, lengths = encode_batch(chains, poses, encoder, config)
            embeddings = embeddings.to(device)
            probabilities = model(embeddings)
            # print(probabilities)
            out = probabilities.squeeze(2).type(torch.float64).cpu()

            loss = loss_fn(out, labels)
            val_loss += loss.data.item()

            all_outs.append(out)
            all_lengths.extend(lengths)
            all_labels.append(labels)

    return val_loss / len(loader), \
        compute_classifier_metrics(torch.cat(all_outs),
                                   torch.cat(all_labels),
                                   all_lengths,
                                   epoch,
                                   file_name,
                                   threshold,
                                   cv,
                                   config)
