import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self):
        """
        Im Konstruktor definieren wir die Bausteine unseres Netzwerks.
        """
        super(SimpleCNN, self).__init__()

        # Erster Convolutional Block
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Nach diesem Block hat unser 28x28 Bild den Shape [16, 14, 14]

        # Zweiter Convolutional Block
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Nach diesem Block hat unser Bild den Shape [32, 7, 7]

        # Klassifikations-Teil (Fully Connected Layer)
        # Wir müssen den 3D-Tensor [32, 7, 7] in einen 1D-Vektor umwandeln (flatten).
        # 32 * 7 * 7 = 1568
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=32 * 7 * 7, out_features=10)  # 10, weil wir 10 Ziffern (0-9) haben
        )

    def forward(self, x):
        """
        In der forward-Methode definieren wir den Datenfluss.
        """
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x


# --- TESTEN WIR UNSER MODELL ---
# Erstellen eine Instanz unseres CNNs
model = SimpleCNN()

# Erstellen einen gefälschten (dummy) Input-Batch mit 64 Bildern
# Shape: [BatchSize, Channels, Height, Width]
dummy_input = torch.randn(64, 1, 28, 28)

# Schicken den Dummy-Input durch das Modell
output = model(dummy_input)

# Überprüfen den Output-Shape
print(f"Input Shape: {dummy_input.shape}")
print(f"Output Shape: {output.shape}")