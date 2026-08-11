import matplotlib.pyplot as plt
import numpy as np

# 1. Hardcoded data extracted from your logs
epochs = list(range(1, 16))
train_loss = [0.5713, 0.4318, 0.3617, 0.2938, 0.2318, 0.1691, 0.1243, 0.0902, 0.0591, 0.0441, 0.0286, 0.0230, 0.0199, 0.0171, 0.0115]
val_acc = [69.24, 73.86, 74.09, 72.42, 76.06, 76.21, 78.94, 78.94, 79.39, 79.39, 79.85, 79.17, 79.32, 80.30, 79.77]

print("Generare grafice de antrenament...")

# 2. Create a figure with 2 side-by-side plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Graph 1: Training Loss
ax1.plot(epochs, train_loss, marker='o', color='red', label='Train Loss')
ax1.set_title("Evoluția Pierderii (Training Loss)")
ax1.set_xlabel("Epocă (Epoch)")
ax1.set_ylabel("Loss")
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend()

# Graph 2: Validation Accuracy
ax2.plot(epochs, val_acc, marker='o', color='blue', label='Validation Accuracy')
ax2.set_title("Evoluția Acurateței pe Validare")
ax2.set_xlabel("Epocă (Epoch)")
ax2.set_ylabel("Acuratețe (%)")
ax2.grid(True, linestyle='--', alpha=0.7)

# 3. Highlight the highest accuracy point (Epoch 14)
best_epoch = np.argmax(val_acc) + 1
ax2.axvline(x=best_epoch, color='green', linestyle='--', label=f'Best Model (Epoch {best_epoch})')
ax2.legend()

plt.tight_layout()
plt.show()