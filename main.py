import os
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
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
import torchvision.transforms as T


df = pd.read_csv("dataset/train_cropped_no_duplicates.csv")

print("Dataset shape: ", df.shape)

# ==========================================
# 1. Unified Dataset Class
# ==========================================
class PneumoniaDetector(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        # Reads the CSV file path directly upon initialization
        self.df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct the file path using the ID column
        patient_id = str(self.df.iloc[idx]['patientId'])
        img_name = f"{patient_id}.png" 
        img_path = os.path.join(self.img_dir, img_name)

        # Load the image using OpenCV
        image = cv2.imread(img_path)
        
        # Safety check to catch missing images
        if image is None:
            raise FileNotFoundError(f"OpenCV could not find or read the image at: {img_path}")
        
        # Convert BGR to RGB for standard plotting/processing
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

        # Extract the label
        label = self.df.iloc[idx]['Target']

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)


selected_features = ["patientId", "Target", "class", "age", "sex"]

train_df, temp_df = train_test_split(
    df,
    test_size=0.40,
    random_state=42,
    stratify=df["Target"]
)

# Second split: Divide Temp evenly into 20% Val, 20% Test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["Target"]
)

# Save the splits to separate CSV files
train_df.to_csv("dataset/train.csv", index=False)
val_df.to_csv("dataset/val.csv", index=False)
test_df.to_csv("dataset/test.csv", index=False)

# ==========================================
# 3. Instantiate the Datasets
# ==========================================
# Use the unified PneumoniaDetector class for all splits
# Make sure the img_dir points to "dataset/New_DS/" to avoid FileNotFoundError

train_transforms = T.Compose([
    T.ToPILImage(),
    T.RandomRotation(degrees=10),
    T.RandomRotation(degrees = 10),
    T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    T.Resize((224, 224)),
    T.ToTensor()
    ])

train_ds = PneumoniaDetector("dataset/train.csv", "dataset/New_DS/", transform=train_transforms)
val_ds   = PneumoniaDetector("dataset/val.csv", "dataset/New_DS/")
test_ds  = PneumoniaDetector("dataset/test.csv", "dataset/New_DS/")

print("--- Dataset Splits ---")
print(f"Training samples: {len(train_ds)}")
print(f"Validation samples: {len(val_ds)}")
print(f"Testing samples: {len(test_ds)}")

# ==========================================
# 4. Test the Pipeline
# ==========================================
print("\n--- Testing Data Loading ---")
# Grab a sample from the training set to verify everything works
test_image, test_label = train_ds[6]

#plt.imshow(test_image)
#plt.title(f"Target is: {test_label.item()} ")
#plt.axis("off")
#plt.show()