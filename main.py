import os
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

df = pd.read_csv("dataset/train_cropped_no_duplicates.csv")

df.head()
df.describe()
print("dataset shape: ", df.shape)


class PneumoniaDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.df = dataframe
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self):
        # Returns the total number of samples in your dataframe
        return len(self.df)
    def __getitem__(self, idx):
        # 1. Construct the file path using the ID column
        # Update 'patientId' if your CSV uses a different column name for the ID
        patient_id = str(self.df.iloc[idx]['patientId'])
        
        # Append the correct file extension (e.g., .jpg, .png)
        img_name = f"{patient_id}.png" 
        img_path = os.path.join(self.img_dir, img_name)

        # 2. Load the image using OpenCV
        image = cv2.imread(img_path)
        
        # OpenCV loads in BGR, so convert to RGB for standard plotting/processing
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

        # 3. Extract the label
        # Update 'Target' if your CSV uses a different column name for the label (e.g., 'class')
        label = self.df.iloc[idx]['Target']

        # 4. Apply any PyTorch transformations (resizing, tensor conversion, etc.)
        if self.transform:
            image = self.transform(image)

        
        return image, torch.tensor(label, dtype=torch.long)


pneumonia_dataset = PneumoniaDataset(dataframe = df, img_dir = "dataset/New_DS/")
test_image, test_label = pneumonia_dataset[100]
plt.imshow(test_image)
plt.title(f"target is: {test_label} ")
plt.axis("off")
plt.show()