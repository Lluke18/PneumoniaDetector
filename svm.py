import os
import pandas as pd
import cv2
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import sys
import os
from torch.utils.data import Subset, Dataset
import torch
from torchvision.io import decode_image
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
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
import joblib



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
        label = self.img_labels.loc[idx, "Target"]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label



transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(),
])

train_ds = CustomImageDataset(
    "./dataset/train.csv",
    "./dataset/New_DS/",
    transform=transform
)

val_ds = CustomImageDataset(
    "./dataset/val.csv",
    "./dataset/New_DS/",
    transform=transform
)

test_ds = CustomImageDataset(
    "./dataset/test.csv",
    "./dataset/New_DS/",
    transform=transform
)



X_train_images = []
y_train_all = []

for image, label in train_ds:
    X_train_images.append(image.numpy())
    y_train_all.append(label)

X_train_images = np.array(X_train_images)
y_train_all = np.array(y_train_all)


print("shape X:", X_train_images.shape)
print("shape y:", y_train_all)


X_val_images = []
y_val_all = []
for image, label in val_ds:
    X_val_images.append(image.numpy())
    y_val_all.append(label)

X_val_images = np.array(X_val_images)
y_val_all = np.array(y_val_all)

print("shape Xval:", X_val_images.shape)


#------------train and val done------------------------

X_test_images = []
y_test_all = []
for image, label in test_ds:
    X_test_images.append(image.numpy())
    y_test_all.append(label)

X_test_images = np.array(X_test_images)
y_test_all = np.array(y_test_all)

#---------------test done------------------------------

X_train_svm = X_train_images.reshape( X_train_images.shape[0], -1)
X_val_svm = X_val_images.reshape( X_val_images.shape[0], -1)

#---------------start SVM

svm = SVC(
    kernel="rbf",
    gamma="scale",
    C=3,
    class_weight="balanced"
)
svm.fit(X_train_svm, y_train_all)

y_val_pred = svm.predict(X_val_svm)

val_accuracy = accuracy_score(y_val_all, y_val_pred)

print("Validation accuracy:", val_accuracy)


#-----------------test-------------------------

X_test_svm = X_test_images.reshape(X_test_images.shape[0], -1)

y_test_pred = svm.predict(X_test_svm)
test_accuracy = accuracy_score(y_test_all, y_test_pred)
print(test_accuracy)

#-----------load model into storage---------

#joblib.dump(svm, "pneumonia_final_svm.pkl") 

#----------------end---------------------

ConfusionMatrixDisplay.from_predictions(
     y_val_all,
     y_val_pred
)

plt.title("Confusion Matrix - Test Set")
plt.show()

#------------confusion matrix--------------


print("Training:")
print(pd.Series(y_train_all).value_counts())

print("\nValidation:")
print(pd.Series(y_val_all).value_counts())

print("\nTest:")
print(pd.Series(y_test_all).value_counts())

#---------------------------plot-----------------


pca = PCA(n_components=2)

X_train_2d = pca.fit_transform(X_train_svm)

plt.figure(figsize=(10, 7))

plt.scatter(
    X_train_2d[y_train_all == 0, 0],
    X_train_2d[y_train_all == 0, 1],
    label="Class 0",
    alpha=0.5
)

plt.scatter(
    X_train_2d[y_train_all == 1, 0],
    X_train_2d[y_train_all == 1, 1],
    label="Class 1",
    alpha=0.5
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("X-ray Dataset - PCA 2D")
plt.legend()
plt.show()








