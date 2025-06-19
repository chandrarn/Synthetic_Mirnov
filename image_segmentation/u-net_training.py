#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 16:56:28 2025
    - Load dataset of images-masks
    - Build U-Net archetecture
    - Run training loop
@author: rianc
"""

# Libraries 
import torch
import torch.nn as nn
import torch.nn.functional as F
import os # Added for path operations
from torchvision import transforms # Import for dataset transforms
from torch.utils.data import Dataset, DataLoader # Import for dataset and dataloader
import numpy as np # Import for dataset generation
from PIL import Image, ImageDraw # Import for dataset generation
import random # Import for dataset generation
from scipy.ndimage import gaussian_filter # Import for dataset generation
import matplotlib.pyplot as plt # Import for visualization

SAVE_PATH = 'dataset.pt'

# --- U-Net Building Blocks (from previous immersive) ---

class DoubleConv(nn.Module):
    """
    Helper function to perform two consecutive convolutional layers.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), # Batch normalization for stability
            nn.ReLU(inplace=True),        # In-place ReLU for memory efficiency
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """
    Downsampling block: MaxPool2d followed by a DoubleConv.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels for the convolutional block.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),          # Halves spatial dimensions
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """
    Upsampling block: Upsampling (ConvTranspose2d) followed by concatenation
    with a skip connection and a DoubleConv.
    Args:
        in_channels (int): Number of input channels (from down path + skip connection).
        out_channels (int): Number of output channels for the convolutional block.
        bilinear (bool): If True, use bilinear interpolation for upsampling.
                         If False, use ConvTranspose2d.
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            # Upsample by 2 using bilinear interpolation (then conv)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels) # No reduction in channels before conv
        else:
            # Transposed convolution for learnable upsampling
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels) # Accounts for concatenated channels

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1) # dim=1 is the channel dimension
        return self.conv(x)

# --- U-Net Model Definition (from previous immersive) ---
class UNet(nn.Module):
    """
    The full U-Net architecture.
    Args:
        n_channels (int): Number of input image channels (e.g., 3 for RGB).
        n_classes (int): Number of output segmentation classes (e.g., 1 for binary mask).
        bilinear (bool): If True, use bilinear interpolation for upsampling.
    """
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder (Downsampling Path)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # Decoder (Upsampling Path)
        factor = 2 if bilinear else 1 

        self.up1 = Up(1024 + 512, 512 // factor, bilinear)
        self.up2 = Up(512 + 256, 256 // factor, bilinear)
        self.up3 = Up(256 + 128, 128 // factor, bilinear)
        self.up4 = Up(128 + 64, 64, bilinear)

        # Output layer (1x1 convolution to map to n_classes)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits

# --- Data Transformation Pipeline (from previous immersive) ---
# For this example, we'll only convert to tensor.
# Transforms are applied inside __getitem__ in SyntheticSegmentationDataset.


try:
    dataset = torch.load(SAVE_PATH, weights_only=False)
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}. Generating new dataset.")

'''
# --- Create Dataset ---
# Check if the dataset already exists and load it, otherwise generate
print(f"Checking for existing dataset at {SAVE_PATH}...")
if os.path.exists(SAVE_PATH):
    try:
        dataset = torch.load(SAVE_PATH)
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading dataset: {e}. Generating new dataset.")
        dataset = SyntheticSegmentationDataset(
            num_samples=NUM_SAMPLES,
            height=IMAGE_HEIGHT,
            width=IMAGE_WIDTH,
            transforms=None # Transforms handled by dataset
        )
        # Save the newly generated dataset
        print(f"Saving newly generated dataset to {SAVE_PATH}...")
        torch.save(dataset, SAVE_PATH)
        print("Dataset saved successfully.")
else:
    print("No existing dataset found. Generating new dataset.")
    dataset = SyntheticSegmentationDataset(
        num_samples=NUM_SAMPLES,
        height=IMAGE_HEIGHT,
        width=IMAGE_WIDTH,
        transforms=None # Transforms handled by dataset
    )
    # Save the newly generated dataset
    print(f"Saving newly generated dataset to {SAVE_PATH}...")
    torch.save(dataset, SAVE_PATH)
    print("Dataset saved successfully.")
'''

# --- Split Dataset into Training and Validation ---
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")


# --- Create DataLoaders ---
BATCH_SIZE = 4 # Example batch size
# Set num_workers > 0 for better performance in a real environment
# For simple scripts or debugging, 0 is fine.
train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0 
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False, # No need to shuffle validation data
    num_workers=0
)


# Define model parameters
INPUT_CHANNELS = 3  # RGB images
OUTPUT_CLASSES = 2  # Binary segmentation: 0 for background, 1 for object.

# Instantiate the U-Net model
model = UNet(n_channels=INPUT_CHANNELS, n_classes=OUTPUT_CLASSES)

print("\nU-Net model created successfully!")
print(model)

# Move model to device (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"\nModel moved to: {device}")

# Define Loss Function (e.g., CrossEntropyLoss for segmentation)
criterion = nn.CrossEntropyLoss()

# Define Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- NEW: Training Loop ---
NUM_EPOCHS = 5 # Number of training epochs

print(f"\n--- Starting Training for {NUM_EPOCHS} Epochs ---")

train_losses = []
val_losses = []

for epoch in range(NUM_EPOCHS):
    model.train() # Set model to training mode
    running_loss = 0.0
    for batch_idx, (images, masks) in enumerate(train_dataloader):
        images = images.to(device)
        masks = masks.to(device) # Masks are LongTensor (0 or 1)
        
        optimizer.zero_grad() # Clear gradients
        
        outputs = model(images) # Forward pass
        
        # Calculate loss
        # Squeeze the mask's channel dimension if CrossEntropyLoss expects (N, H, W)
        loss = criterion(outputs, masks.squeeze(1)) 
        
        loss.backward() # Backward pass
        optimizer.step() # Update weights

        running_loss += loss.item() * images.size(0) # Accumulate loss

    epoch_train_loss = running_loss / len(train_dataloader.dataset)
    train_losses.append(epoch_train_loss)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {epoch_train_loss:.4f}")

    # --- Validation Loop ---
    model.eval() # Set model to evaluation mode
    val_running_loss = 0.0
    with torch.no_grad(): # Disable gradient calculations for validation
        for images, masks in val_dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks.squeeze(1))
            val_running_loss += loss.item() * images.size(0)

    epoch_val_loss = val_running_loss / len(val_dataloader.dataset)
    val_losses.append(epoch_val_loss)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Validation Loss: {epoch_val_loss:.4f}")

print("\n--- Training Complete ---")

# --- Visualize Training and Validation Loss ---
plt.figure(figsize=(10, 5))
plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label='Training Loss')
plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()


# --- Example Prediction on a Validation Sample ---
print("\n--- Example Prediction on a Validation Sample ---")
model.eval() # Set model to evaluation mode
with torch.no_grad():
    # Get one sample from the validation dataset
    sample_image, sample_mask = val_dataset[0] 
    
    # Add batch dimension and move to device
    sample_image_input = sample_image.unsqueeze(0).to(device)
    
    # Get model prediction
    predicted_logits = model(sample_image_input)
    
    # Get the predicted class mask (argmax over channels)
    predicted_mask = torch.argmax(predicted_logits.squeeze(0), dim=0) # Remove batch dim, get argmax
    
    # Detach from GPU and convert to numpy for visualization
    sample_image_np = sample_image.cpu().permute(1, 2, 0).numpy()
    sample_mask_np = sample_mask.cpu().squeeze().numpy()
    predicted_mask_np = predicted_mask.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Validation Sample: Input, True Mask, Predicted Mask')

    axes[0].imshow(sample_image_np)
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    axes[1].imshow(sample_mask_np, cmap='gray')
    axes[1].set_title('True Mask')
    axes[1].axis('off')

    axes[2].imshow(predicted_mask_np, cmap='gray')
    axes[2].set_title('Predicted Mask')
    axes[2].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

print("\nScript execution complete.")
