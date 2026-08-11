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
import torchvision.models as models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch.nn as nn
import torch.nn.functional as F

df = pd.read_csv("dataset/train_cropped_no_duplicates.csv")

print("Dataset shape: ", df.shape)


class PneumoniaDetector(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        
        self.df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        patient_id = str(self.df.iloc[idx]['patientId'])
        img_name = f"{patient_id}.png" 
        img_path = os.path.join(self.img_dir, img_name)

        
        image = cv2.imread(img_path)
        
        
        if image is None:
            raise FileNotFoundError(f"OpenCV could not find or read the image at: {img_path}")
        
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

        
        label = self.df.iloc[idx]['Target']

        
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


val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["Target"]
)


train_df.to_csv("dataset/train.csv", index=False)
val_df.to_csv("dataset/val.csv", index=False)
test_df.to_csv("dataset/test.csv", index=False)



train_transforms = T.Compose([
    T.ToPILImage(),
    T.RandomRotation(degrees=10),
    T.RandomRotation(degrees = 10),
    T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    T.Resize((224, 224)),
    T.ToTensor()
    ])

train_ds = PneumoniaDetector("dataset/train.csv", "dataset/New_DS/")
val_ds   = PneumoniaDetector("dataset/val.csv", "dataset/New_DS/")
test_ds  = PneumoniaDetector("dataset/test.csv", "dataset/New_DS/")

print("--- Dataset Splits ---")
print(f"Training samples: {len(train_ds)}")
print(f"Validation samples: {len(val_ds)}")
print(f"Testing samples: {len(test_ds)}")


print("Test the model RESNET")

test_image, test_label = train_ds[6]
plt.imshow(test_image)
plt.title(f"Target is: {"Sanatos" if test_label.item() == 0 else "Pneumonie"} ")
plt.axis("off")
plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_loader = DataLoader(test_ds, batch_size = 32, shuffle = False)

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

model.load_state_dict(torch.load("resnet18_best_pneumonia.pth", map_location = device))

model.eval()

test_correct = 0
test_total = 0
all_true_labels = []
all_predicted_labels = []


with torch.no_grad():
    for test_images, test_labels in test_loader:

        test_images = test_images.float() / 255.0
        test_images = test_images.permute(0, 3, 1, 2)
        test_images = F.interpolate(test_images, size=(224, 224), mode="bilinear", align_corners=False)

        test_images = test_images.to(device)
        test_labels = test_labels.to(device)
        
        test_outputs = model(test_images)
        _, test_predicted = torch.max(test_outputs, 1)
        
        test_total += test_labels.size(0)
        test_correct += (test_predicted == test_labels).sum().item()

        all_true_labels.extend(test_labels.cpu().numpy())
        all_predicted_labels.extend(test_predicted.cpu().numpy())

final_test_accuracy = (test_correct / test_total) * 100
print(f"Rezultatul final este: {final_test_accuracy}")

cm = confusion_matrix(all_true_labels, all_predicted_labels)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, 
    display_labels=["Sănătos", "Pneumonie"] 
)
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(cmap=plt.cm.Purples, ax=ax)

plt.title("Matricea de confuzie pentru modelul CNN:", fontsize=14)
plt.show()