import sys
import os
from torch.utils.data import Subset, Dataset
import torch
from torchvision.io import decode_image
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import make_blobs, make_circles
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0] + ".png")
        image = decode_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


train_ds = CustomImageDataset(
    "train_cropped_no_duplicates.csv",
    "./New_DS/"
)

y_all = train_ds.img_labels["Target"].to_numpy()
all_indices = np.arange(len(train_ds))

#--------------------------------------------------------

df = pd.read_csv("train_cropped_no_duplicates.csv")


train_df, temp_df = train_test_split(
    df,
    test_size=0.40,
    random_state=42,
    stratify=df["Target"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["Target"]
)

train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)


train_ds = CustomImageDataset("train.csv", "./New_DS/")
val_ds   = CustomImageDataset("val.csv", "./New_DS/")
test_ds  = CustomImageDataset("test.csv", "./New_DS/")

