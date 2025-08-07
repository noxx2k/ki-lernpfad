import torch
import torch.nn as nn # nn ist das Modul, das alle Bausteine für neuronale Netze enthält
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

# --- 1. Daten laden ---
# Wir laden das MNIST-Trainingsset. `transform=ToTensor()` konvertiert die Bilder in PyTorch-Tensoren.
train_data = MNIST(root='data', train=True, download=True, transform=ToTensor())
# Wir nehmen nur das allererste Bild und sein Label aus dem Datensatz.
image, label = train_data[0]

print(f"Original Shape des Bildes: {image.shape}")
# Wichtiger Hinweis: PyTorch erwartet Batches von Bildern.
# Der Shape ist [ColorChannels, Height, Width].
# Wir fügen eine "Batch-Dimension" von 1 hinzu, um ein einzelnes Bild zu verarbeiten.
# Neuer Shape: [BatchSize, ColorChannels, Height, Width]
image_batch = image.unsqueeze(0)
print(f"Shape nach Hinzufügen der Batch-Dimension: {image_batch.shape}")

# --- 2. CNN-Schichten definieren ---
# Eine Convolutional Layer.
# in_channels=1 (unser Bild hat 1 Farbkanal, da es schwarz-weiß ist)
# out_channels=16 (die Schicht soll 16 verschiedene Feature Maps / Muster erzeugen)
# kernel_size=3 (die "Lupe" ist 3x3 Pixel groß)
conv_layer = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)

# Eine Max-Pooling-Schicht.
# kernel_size=2 (sie schaut sich 2x2 Blöcke an)
# stride=2 (sie springt in 2-Pixel-Schritten)
pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)

# --- 3. Bild durch die Schichten schicken ---
# Bild durch die Convolutional Layer
conv_output = conv_layer(image_batch)
print(f"\nShape nach der Convolutional Layer: {conv_output.shape}")

# Ergebnis durch die Pooling Layer
pool_output = pool_layer(conv_output)
print(f"Shape nach der Pooling Layer: {pool_output.shape}")