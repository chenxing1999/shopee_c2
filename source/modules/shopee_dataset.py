from torch.utils.data import Dataset
from torchvision import transforms

import os
from PIL import Image
import torch

default_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


class ShopeeDataset(Dataset):
    def __init__(self, df, folder_dir, transform=None, num_classes=42):
        self.df = df
        if transform is None:
            transform = default_transform
        self.transform = transform
        self.folder_dir = folder_dir
        self.num_classes = num_classes

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row["filename"]
        y = row["category"]

        if int(y) != self.num_classes:
            fpath = os.path.join(self.folder_dir, str(y).zfill(2), fname)
        else:
            fpath = os.path.join(self.folder_dir, fname)

        img = Image.open(fpath).convert("RGB")

        img = self.transform(img)
        y = int(y)
        return img, torch.tensor(y)
