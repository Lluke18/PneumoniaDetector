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
import torchvision.models as models
from sklearn.utils.class_weight import compute_class_weight



def relu(z):
    
    return np.maximum(0, z)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
print(model)


num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

model = model.to(device)

learning_rate = 0.00001
train_loader = DataLoader(train_ds, batch_size = 32, shuffle = True)
val_loader = DataLoader(val_ds, batch_size = 32, shuffle = False)

train_labels = train_ds.df['Target'].values
class_weights = compute_class_weight(class_weight="balanced", classes = np.unique(train_labels), y = train_labels)
manual_weights = [1.0, 5.0]

weights_tensor = torch.tensor(manual_weights, dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=weights_tensor) # in curs este nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr =learning_rate)


num_epochs = 15
best_val_accuracy = 0.0

print("incepe antrenamentul cu tot cu validare")
time.sleep(2)

#for the graph:
train_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):
        
        images = images.float() / 255.0
        images = images.permute(0, 3, 1, 2)
        images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward() 

        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f" Epoch {epoch+1} Completed. Average Loss: {avg_loss:.4f} ")
  
    model.eval()
    correct_predictions = 0
    total_samples = 0

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

    val_accuracy = (correct_predictions / total_samples) * 100
    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_loss:.4f} | Val Acc: {val_accuracy:.2f}%")
    if val_accuracy > best_val_accuracy:
        print(f" -> Nou scor maxim! Se salvează modelul (Val Acc a crescut de la {best_val_accuracy:.2f}% la {val_accuracy:.2f}%).")
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), "resnet18_secondtry_pneumonia.pth")

print("s-au salvat weightsurile RESNET")

