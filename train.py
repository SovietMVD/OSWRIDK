import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import config
from dataset import PairedDataset
from dbn import DBN


def _device():
    return torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")


def _split(samples, val_frac, seed):
    labels = [l for _, l in samples]
    train_idx, val_idx = train_test_split(
        np.arange(len(samples)), test_size=val_frac, stratify=labels, random_state=seed
    )
    train = [samples[i] for i in train_idx]
    val = [samples[i] for i in val_idx]
    return train, val


def _run_epoch(model, loader, device, optimizer=None, criterion=None):
    if optimizer is not None:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    total = 0
    for gf, lf, y in loader:
        gf = gf.to(device)
        lf = lf.to(device)
        y = y.to(device)
        if optimizer is not None:
            optimizer.zero_grad()
        logits, _ = model(gf, lf)
        loss = criterion(logits, y)
        if optimizer is not None:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * y.size(0)
        total += y.size(0)
    return total_loss / total


def train(samples, num_classes, out_path, seed=None):
    seed = config.SEED if seed is None else seed
    device = _device()

    train_samples, val_samples = _split(samples, config.VAL_FRAC, seed)
    train_loader = DataLoader(
        PairedDataset(train_samples, config.INPUT_SIZE),
        batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        PairedDataset(val_samples, config.INPUT_SIZE),
        batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0,
    )

    model = DBN(config.BACKBONE, num_classes=num_classes, pretrained=config.PRETRAINED)
    model.to(device)
    optimizer = optim.SGD(
        model.parameters(), lr=config.LEARNING_RATE,
        momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss()

    best_val = float("inf")
    best_state = None
    patience = 0
    for _ in range(config.EPOCHS):
        _run_epoch(model, train_loader, device, optimizer, criterion)
        val_loss = _run_epoch(model, val_loader, device, None, criterion)
        if best_val - val_loss > config.DELTA:
            best_val = val_loss
            best_state = model.state_dict()
            patience = 0
        else:
            patience += 1
        if patience >= config.PATIENCE:
            break

    if best_state is None:
        best_state = model.state_dict()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(best_state, out_path)
    return model
