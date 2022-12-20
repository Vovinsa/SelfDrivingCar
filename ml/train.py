import torch
from torch.utils.data import DataLoader

import pandas as pd

from dataset import CarDataset
from models import seresnet18

import argparse
import logging


parser = argparse.ArgumentParser(description="Train parser")
parser.add_argument("--imgs_path", type=str,
                    help="Path to the images")
parser.add_argument("--csv_train_path", type=str,
                    help="Path to train csv file, which consists of images path and targets")
parser.add_argument("--csv_validation_path", type=str,
                    help="Path to validation csv file, which consists of images path and targets", default=None)
parser.add_argument("--lr", type=float,
                    help="Learning rate", default=1e-3)
parser.add_argument("--batch_size", type=int,
                    help="Batch size", default=32)
parser.add_argument("--epochs", type=int,
                    help="Epochs count for train", default=10)


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        img, rot_angle, command = batch
        img, rot_angle, command = img.to(device), rot_angle.to(device), command.to(device)
        preds = model(img, rot_angle, command)
        loss = criterion(preds, rot_angle)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    valid_loss = 0
    for batch in dataloader:
        img, rot_angle, command = batch
        img, rot_angle, command = img.to(device), rot_angle.to(device), command.to(device)
        preds = model(img, rot_angle, command)
        loss = criterion(preds, rot_angle)
        valid_loss += loss.item()
    return valid_loss


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    args = parser.parse_args()

    train_df = pd.read_csv(args.csv_train_path)
    train_dataset = CarDataset(train_df)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=False)

    validation_df = pd.read_csv(args.csv_validation_path)
    validation_dataset = CarDataset(validation_df)
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=False)

    model = seresnet18.make_seresnet18()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = torch.nn.MSELoss(reduction="mean")

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_dataloader, optim, crit, DEVICE)
        validation_loss = validate_epoch(model, validation_dataloader, crit, DEVICE)
        torch.save(model.state_dict(), f"weights/{epoch}-{train_loss}.pth")
