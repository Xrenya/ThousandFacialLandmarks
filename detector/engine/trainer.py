from typing import List

from hydra.utils import instantiate
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import DataParallel

from detector.utils.utils import set_global_seed, get_gpus_ids, data_loader, optimizer
from detector.data.dataset import get_loaders
import numpy as np


def train(config):
    if config.seed  is not None:
        set_global_seed(config.seed)

    device_ids = get_gpus_ids(config.num_gpus)
    device = torch.device("cuda" if device_ids else "cpu")
    model = instantiate(config.model)
    model = DataParallel(model, device_ids)
    model.to(device)
    train_loader, valid_loader, test_loader = get_loaders(config)
    best_val_loss = np.inf

    # for epoch in range(config.epochs):
    #     train_loss = train_on_epoch(model, train_loader, loss_fn, optimizer, device=device)
    #     val_loss = validate_on_ecpoh(model, valid_loader, loss_fn, device=device)
    #     print("Epoch #{:2}:\ttrain loss: {:5.2}\tval loss: {:5.2}".format(epoch, train_loss, val_loss))
    #     if val_loss < best_val_loss:
    #         best_val_loss = val_loss
    #         with open(os.path.join("runs", f"{args.name}_best.pth"), "wb") as fp:
    #             torch.save(model.state_dict(), fp)
    #train_data, val_data = data_loader(config.data)

    # loss = instantiate(config.loss)
    # loss.to(device)

    # optimizer =
    return 0

def train_on_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    train_loss = []
    for batch in tqdm.tqdm(loader, total=len(loader), desc="training..."):
        images = batch["image"].to(device)  # B x 3 x CROP_SIZE x CROP_SIZE
        landmarks = batch["landmarks"]  # B x (2 * NUM_PTS)

        pred_landmarks = model(images).cpu()  # B x (2 * NUM_PTS)
        loss = loss_fn(pred_landmarks, landmarks, reduction="mean")
        train_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.mean(train_loss)


def validate_on_ecpoh(model, loader, loss_fn, device):
    model.eval()
    val_loss = []
    for batch in tqdm.tqdm(loader, total=len(loader), desc="validation..."):
        images = batch["image"].to(device)
        landmarks = batch["landmarks"]

        with torch.no_grad():
            pred_landmarks = model(images).cpu()
        loss = loss_fn(pred_landmarks, landmarks, reduction="mean")
        val_loss.append(loss.item())

    return np.mean(val_loss)
