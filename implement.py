# Install required libraries
!pip install torch torchvision matplotlib

# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.quantization import quantize_dynamic
import numpy as np
import random

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
batch_size = 16
num_epochs = 20
learning_rate = 0.0002
image_size = 64
mask_ratio = 0.25  # Central mask ratio

# Data Preprocessing
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])

# Load CIFAR-10 dataset
dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Generate random masks for training
def create_mask(batch_size, img_size, mask_ratio):
    mask = torch.ones((batch_size, 1, img_size, img_size))
    mask_start = int(img_size * (1 - mask_ratio) / 2)
    mask_end = int(img_size * (1 + mask_ratio) / 2)
    mask[:, :, mask_start:mask_end, mask_start:mask_end] = 0
    return mask.to(device)

# Define the Dual-Branch Generator Network
class DualBranchGenerator(nn.Module):
    def __init__(self):
        super(DualBranchGenerator, self).__init__()

        # Encoder: Extract features
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        )

        # Decoder: Reconstruct the image
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1), nn.Tanh(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed

# Apply Model Pruning
def apply_pruning(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
            torch.nn.utils.prune.l1_unstructured(module, name="weight", amount=0.2)
    return model

# Loss Function (Masked L1 Loss)
def masked_loss(reconstructed, original, mask):
    loss = torch.mean(torch.abs(reconstructed - original) * (1 - mask))
    return loss

# Initialize model and optimizer
generator = DualBranchGenerator().to(device)

# Quantization before Pruning
generator = quantize_dynamic(generator, {nn.Conv2d, nn.ConvTranspose2d}, dtype=torch.qint8)  # Quantization

generator = apply_pruning(generator)  # Pruning
optimizer = optim.Adam(generator.parameters(), lr=learning_rate)



# Training Loop
def train(generator, dataloader, num_epochs):
    generator.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for images, _ in dataloader:
            images = images.to(device)
            mask = create_mask(images.size(0), image_size, mask_ratio)
            masked_images = images * mask

            # Forward pass
            reconstructed = generator(masked_images)
            loss = masked_loss(reconstructed, images, mask)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}")

# Train the model
train(generator, dataloader, num_epochs)

# Visualize Results
def visualize_results(generator, dataloader):
    generator.eval()
    with torch.no_grad():
        for images, _ in dataloader:
            images = images[:8].to(device)
            mask = create_mask(images.size(0), image_size, mask_ratio)
            masked_images = images * mask
            reconstructed = generator(masked_images)

            # Unnormalize for visualization
            images = (images + 1) / 2
            masked_images = (masked_images + 1) / 2
            reconstructed = (reconstructed + 1) / 2

            # Display results
            grid = torch.cat([images, masked_images, reconstructed], dim=0)
            grid = utils.make_grid(grid, nrow=8)
            plt.figure(figsize=(20, 10))
            plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
            plt.axis('off')
            plt.show()
            break

# Visualize the inpainting results
visualize_results(generator, dataloader)
