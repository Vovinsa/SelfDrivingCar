import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import mlflow

import pandas as pd

from utils.dataset import CarDataset
from models.branched_network import BranchedNetwork

import argparse
import logging
import os


mlflow.set_tracking_uri("http://192.168.100.9:5001")

logger = logging.getLogger("train_self_driving")
logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Train parser")
parser.add_argument("--imgs_path", type=str,
                    help="Path to the images", default="data/images")
parser.add_argument("--csv_train_path", type=str,
                    help="Path to train csv file, which consists of images path and targets", default="data/train.csv")
parser.add_argument("--csv_validation_path", type=str,
                    help="Path to validation csv file, which consists of images path and targets", default="data/train.csv")
parser.add_argument("--lr", type=float,
                    help="Learning rate", default=1e-3)
parser.add_argument("--batch_size", type=int,
                    help="Batch size", default=32)
parser.add_argument("--epochs", type=int,
                    help="Epochs count for train", default=10)
parser.add_argument("--experiment_name", type=str,
                    help="MLFlow experiment name", default="default")
parser.add_argument("--run_name", type=str,
                    help="MLFlow run name", default="default")


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        img, measurements = batch
        img, measurements = img.to(device), measurements.to(device)
        preds = model(img)
        preds = torch.stack(preds, dim=1).to(device)
        loss = criterion(preds, measurements)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    valid_loss = 0
    for batch in dataloader:
        img, measurements = batch
        img, measurements = img.to(device), measurements.to(device)
        preds = model(img)
        preds = torch.stack(preds, dim=1).to(device)
        loss = criterion(preds, measurements)
        valid_loss += loss.item()
    return valid_loss / len(dataloader)


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    try:
        train_df = pd.read_csv(args.csv_train_path)
    except Exception as e:
        logger.exception(f"Unable to open {args.csv_train_path}")
    train_dataset = CarDataset(train_df, "data/images", transform)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=False)

    try:
        validation_df = pd.read_csv(args.csv_validation_path)
    except Exception as e:
        logger.exception(f"Unable to open {args.csv_validation_path}")
    validation_dataset = CarDataset(validation_df, "data/images", transform)
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=False)

    model = BranchedNetwork(emb_size=128).to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = torch.nn.HuberLoss(reduction="mean")

    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=args.run_name):
        if not os.path.exists("weights"):
            os.mkdir("weights")
            logger.info("Create dir weights")
        if not os.path.exists(f"weights/{args.experiment_name}"):
            os.mkdir(f"weights/{args.experiment_name}")
            logger.info(f"Create dir weights/{args.experiment_name}")
        if not os.path.exists(f"weights/{args.experiment_name}/{args.run_name}"):
            os.mkdir(f"weights/{args.experiment_name}/{args.run_name}")
            logger.info(f"Create dir weights/{args.experiment_name}/{args.run_name}")
        for epoch in range(args.epochs):
            logger.info(f"Epoch {epoch + 1}")
            train_loss = train_epoch(model, train_dataloader, optim, crit, DEVICE)
            validation_loss = validate_epoch(model, validation_dataloader, crit, DEVICE)
            mlflow.log_metric("Train loss", train_loss, step=epoch+1)
        mlflow.log_param("lr", args.lr)
        mlflow.log_param("epochs", args.epochs)
        torch.save(model.state_dict(), f"weights/{args.experiment_name}/{args.run_name}/loss-{validation_loss}.pth")
