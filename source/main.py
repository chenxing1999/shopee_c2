from modules import SimpleModule, ShopeeDataset
from pytorch_lightning.utilities.seed import seed_everything

seed_everything(1)

import argparse
import os
import numpy as np
import torch
from torch import nn
from PIL import Image

from sklearn.model_selection import train_test_split
import pandas as pd
import pytorch_lightning as pl

from torchvision import models
from torch.utils.data import DataLoader
import datetime


root_folder = os.path.dirname(__file__)
default_train = "../data/train.csv"
default_train = os.path.join(root_folder, default_train)

default_test = "../data/test.csv"
default_test = os.path.join(root_folder, default_test)


def parse_args():
    """ Load hyper parameters helper """
    parser = argparse.ArgumentParser()
    parser.add_argument("train_folder", help="Path to train folder")
    parser.add_argument("--train_csv", default=default_train)
    parser.add_argument("--val_csv", default=None)

    parser.add_argument("--test_folder", default=None)
    parser.add_argument("--test_csv", default=default_test)

    parser.add_argument("--num_classes", default=42)
    parser.add_argument("--batch_size", default=32)
    parser.add_argument("--logdir", default=None)
    parser.add_argument("--lr", default=1e-2)
    parser.add_argument("--weight_path", default=None)

    return parser.parse_args()


def inference(
    model,
    test_df,
    test_folder,
    transform,
    output_path=None,
    device="cuda",
):
    print("Start inference")
    model.eval()
    model.to(device)

    if output_path is None:
        ts = datetime.datetime.now().strftime("%d%m-%H%M%S")
        output_path = os.path.join(f"output/output_{ts}.txt")

    fout = open(output_path, "w")
    fout.write("filename,category\n")
    for i, row in test_df.iterrows():
        fname = row["filename"]
        path = os.path.join(test_folder, fname)
        img = Image.open(path)
        x = transform(img)
        x = x.unsqueeze(0)
        x = x.to(device)
        y_hat = model(x)
        labels_hat = torch.argmax(y_hat, dim=1).item()

        if i % 1000 == 0:
            print(i)
        fout.write(f"{fname},{labels_hat}\n")


def main():
    args = parse_args()

    # Load data
    train_df = pd.read_csv(args.train_csv)
    # Shuffle
    train_df = train_df.sample(frac=1)

    if args.val_csv is None:
        train_df, val_df = train_test_split(
            train_df,
            test_size=0.2,
            random_state=0,
            stratify=train_df["category"],
        )
    else:
        val_df = pd.read_csv(args.val_csv)

    train_dataset = ShopeeDataset(train_df, args.train_folder)
    val_dataset = ShopeeDataset(val_df, args.train_folder)

    batch_size = int(args.batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)

    # Load model
    # model = models.wide_resnet50_2(True)
    # model.fc = nn.Linear(model.fc.in_features, int(args.num_classes))
    from efficientnet_pytorch import EfficientNet
    model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=int(args.num_classes))

    lr = float(args.lr)
    module = SimpleModule(model, lr)
    trainer = pl.Trainer(max_epochs=100, gpus=1, early_stop_callback=False)
    if args.weight_path:
        checkpoint = torch.load(args.weight_path)
        #module.load_from_checkpoint(checkpoint_path=args.weight_path)
        module.load_state_dict(checkpoint["state_dict"])
        # trainer.load_from_checkpoint(checkpoint_path=args.weight_path)


    trainer.fit(module, train_loader, val_loader)

    test_df = pd.read_csv(args.test_csv)
    inference(module, test_df, args.test_folder, train_dataset.transform)


if __name__ == "__main__":
    main()
