import numpy as np
from main import PneumoniaDetector
from main import train_ds
from main import val_ds
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import time
import torch.nn.functional as F
from torch.utils.data import DataLoader


def relu(z):
    
    return np.maximum(0, z)

def leaky_relu(z, alpha=0.1): # Folosim alpha=0.1 pentru a face linia mai vizibilă pe grafic
    return np.where(z > 0, z, alpha * z)

def tanh(z):
    return np.tanh(z)

class PneumoniaCNN(nn.Module):
    def __init__(self, in_features, num_classes):
        super(PneumoniaCNN, self).__init__()

        self.layer1 = nn.Conv2d(in_channels=in_features, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        self.layer2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=64 * 28 * 28, out_features=128)
        self.dropout = nn.Dropout(p=0.5) # Helps prevent overfitting

        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x =self.pool1(x)
        x = self.relu(self.layer2(x))
        x = self.pool2(x)
        x = self.relu(self.layer3(x))
        x = self.pool3(x)

        x = torch.flatten(x, 1)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = PneumoniaCNN(3, 2).to(device)
model.load_state_dict(torch.load("pneumonia_cnn_weights.pth", map_location=device))
print("Weights successfully loaded!")
print(model)

#dummy_images = torch.randn(4, 3, 224, 224).to(device)


#output = model(dummy_images)
#print(f"Output shape: {output.shape}")

#Now, we optimize
learning_rate = 0.0001
train_loader = DataLoader(train_ds, batch_size = 32, shuffle = True)

criterion = nn.CrossEntropyLoss() # in course, it is: nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr =learning_rate)


num_epochs = 30

print("incepe antrenamentul")
time.sleep(2)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):
        
        #images = images.float() / 255.0
        #images = images.permute(0, 3, 1, 2)
        #images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward() # BACKWARD PASS

        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f" Epoch {epoch+1} Completed. Average Loss: {avg_loss:.4f} ")

print("S-a terminat antrenamentul")

torch.save(model.state_dict(), "pneumonia_cnn_weights.pth")
print("s-au salvat weightsurile")


val_loader = DataLoader(val_ds, batch_size = 32, shuffle = False)

model.eval()

correct_predictions = 0
total_samples = 0

print("\nÎncepe evaluarea pe setul de validare...")


with torch.no_grad():
    for images, labels in val_loader:
        
        images = images.float() / 255.0
        images = images.permute(0, 3, 1, 2)
        images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)

        images = images.to(device)
        labels = labels.to(device)

        
        outputs = model(images)
        
        
        _, predicted_class = torch.max(outputs, 1)

        total_samples += labels.size(0)
        correct_predictions += (predicted_class == labels).sum().item()

accuracy = (correct_predictions / total_samples)
print(f"final acc pe valdiare este: {accuracy:.2f}%")