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

# --- Configuration Parameters (from previous immersive, needed for dataset) ---
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
NUM_SAMPLES = 200 # Increased for more meaningful training
SAVE_PATH = 'dataset.pt'#'synthetic_segmentation_dataset.pth' # Path to save/load dataset

# Shape generation parameters (from previous immersive)
NUM_SHAPES_PER_IMAGE = (2, 5)
SHAPE_SIZE_RANGE = (10, 40)
SHAPE_INTENSITY_RANGE = (100, 255)

# Squiggle generation parameters (from previous immersive)
NUM_SQUIGGLES_PER_IMAGE = (1, 3)
SQUIGGLE_SEGMENTS = (5, 15)
SQUIGGLE_STEP_SIZE = (5, 20)
SQUIGGLE_LINE_WIDTH = (1, 5)
SQUIGGLE_INTENSITY_RANGE = (50, 200)

# Background intensity parameters (from previous immersive)
BACKGROUND_NOISE_SCALE = 0.05
BACKGROUND_BLUR_SIGMA = 15

# --- Helper Function for Image and Mask Generation (from previous immersive) ---
def generate_image_and_mask(height, width):
    """
    Generates a single synthetic image with random shapes/squiggles and a corresponding
    binary mask containing bounding boxes around the generated objects.
    """
    low_res_h, low_res_w = height // 8, width // 8
    base_intensity = np.random.rand(low_res_h, low_res_w) * 255 * BACKGROUND_NOISE_SCALE
    base_intensity = gaussian_filter(base_intensity, sigma=BACKGROUND_BLUR_SIGMA / 8)
    base_intensity = np.interp(base_intensity, (base_intensity.min(), base_intensity.max()), (50, 150))
    
    base_image_pil = Image.fromarray(base_intensity.astype(np.uint8)).resize((width, height), Image.Resampling.BICUBIC)
    image = np.array(base_image_pil)
    image = np.stack([image, image, image], axis=-1)
    image = Image.fromarray(image.astype(np.uint8))
    
    draw = ImageDraw.Draw(image)

    mask = Image.new('L', (width, height), 0)
    mask_draw = ImageDraw.Draw(mask)

    all_bounding_boxes = []

    num_shapes = random.randint(*NUM_SHAPES_PER_IMAGE)
    for _ in range(num_shapes):
        shape_type = random.choice(['circle', 'rectangle'])
        size = random.randint(*SHAPE_SIZE_RANGE)
        
        x = random.randint(0, width - size)
        y = random.randint(0, height - size)
        
        intensity = random.randint(*SHAPE_INTENSITY_RANGE)
        shape_color = (intensity, intensity, intensity)

        if shape_type == 'circle':
            bbox = [x, y, x + size, y + size]
            draw.ellipse(bbox, fill=shape_color)
        else:
            bbox = [x, y, x + size, y + size]
            draw.rectangle(bbox, fill=shape_color)
        
        all_bounding_boxes.append(bbox)

    num_squiggles = random.randint(*NUM_SQUIGGLES_PER_IMAGE)
    for _ in range(num_squiggles):
        num_segments = random.randint(*SQUIGGLE_SEGMENTS)
        start_x = random.randint(0, width)
        start_y = random.randint(0, height)
        points = [(start_x, start_y)]
        
        current_x, current_y = start_x, start_y
        min_x, max_x = current_x, current_x
        min_y, max_y = current_y, current_y

        for _ in range(num_segments - 1):
            step_x = random.randint(-SQUIGGLE_STEP_SIZE[1], SQUIGGLE_STEP_SIZE[1])
            step_y = random.randint(-SQUIGGLE_STEP_SIZE[1], SQUIGGLE_STEP_SIZE[1])
            
            next_x = max(0, min(width - 1, current_x + step_x))
            next_y = max(0, min(height - 1, current_y + step_y))
            
            points.append((next_x, next_y))
            current_x, current_y = next_x, next_y

            min_x = min(min_x, current_x)
            max_x = max(max_x, current_x)
            min_y = min(min_y, current_y)
            max_y = max(max_y, current_y)

        line_width = random.randint(*SQUIGGLE_LINE_WIDTH)
        squiggle_intensity = random.randint(*SQUIGGLE_INTENSITY_RANGE)
        squiggle_color = (squiggle_intensity, squiggle_intensity, squiggle_intensity)

        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i+1]
            draw.line([p1, p2], fill=squiggle_color, width=line_width)
        
        bbox_squiggle = [
            max(0, min_x - line_width),
            max(0, min_y - line_width),
            min(width, max_x + line_width),
            min(height, max_y + line_width)
        ]
        all_bounding_boxes.append(bbox_squiggle)

    for bbox in all_bounding_boxes:
        mask_draw.rectangle(bbox, fill=255)

    return image, mask

# --- PyTorch Dataset Class (from previous immersive) ---
class SyntheticSegmentationDataset(Dataset):
    def __init__(self, num_samples, height, width, transforms=None):
        super().__init__() # Call parent constructor
        self.num_samples = num_samples
        self.height = height
        self.width = width
        self.transforms = transforms
        self.data = []
        for i in range(num_samples):
            img, mask = generate_image_and_mask(self.height, self.width)
            self.data.append((img, mask))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image, mask = self.data[idx]

        if self.transforms:
            image = transforms.ToTensor()(image)
            mask = transforms.ToTensor()(mask)
            mask = (mask * 255).to(torch.long)
            mask = (mask > 0).to(torch.long) # Convert to binary 0 or 1
        return image, mask

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
# --- PyTorch Dataset Class (from previous immersive) ---
class SyntheticSegmentationDataset(Dataset):
    def __init__(self, num_samples, height, width, transforms=None):
        super().__init__() # Call parent constructor
        self.num_samples = num_samples
        self.height = height
        self.width = width
        self.transforms = transforms
        self.data = []
        for i in range(num_samples):
            img, mask = generate_image_and_mask(self.height, self.width)
            self.data.append((img, mask))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image, mask = self.data[idx]

        if self.transforms:
            image = transforms.ToTensor()(image)
            mask = transforms.ToTensor()(mask)
            mask = (mask * 255).to(torch.long)
            mask = (mask > 0).to(torch.long) # Convert to binary 0 or 1
        return image, mask

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
        # Fix: The out_channels of Up blocks should match the channels of the skip connection from the *previous* level
        # so that the output matches the corresponding encoder stage when going up.
        # This is where the channels are effectively halved after concatenation in typical U-Net implementations.
        self.up1 = Up(1024 + 512, 512, bilinear) # Concatenated input: 1536. Output of DoubleConv: 512.
        self.up2 = Up(512 + 256, 256, bilinear)   # Concatenated input: 768. Output of DoubleConv: 256.
        self.up3 = Up(256 + 128, 128, bilinear)   # Concatenated input: 384. Output of DoubleConv: 128.
        self.up4 = Up(128 + 64, 64, bilinear)     # Concatenated input: 192. Output of DoubleConv: 64.

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

# --- Create Dataset ---
# Check if the dataset already exists and load it, otherwise generate
print(f"Checking for existing dataset at {SAVE_PATH}...")
if True:#os.path.exists(SAVE_PATH):
    try:
        dataset = torch.load(SAVE_PATH,weights_only=False)
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading dataset: {e}. Generating new dataset.")
        raise SyntaxError
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
        transforms=True # Transforms handled by dataset
    )
    # Save the newly generated dataset
    print(f"Saving newly generated dataset to {SAVE_PATH}...")
    torch.save(dataset, SAVE_PATH)
    print("Dataset saved successfully.")

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
    num_workers=12 
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False, # No need to shuffle validation data
    num_workers=12
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
NUM_EPOCHS = 10 # Number of training epochs

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
