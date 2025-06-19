import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from PIL import Image, ImageDraw
import random
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from multiprocessing import cpu_count

# --- Configuration Parameters ---
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
NUM_SAMPLES = 50 # Increased samples for better regression training
SAVE_PATH = 'synthetic_multi_bbox_dataset.pth' # New save path for multi-bbox dataset

# Shape generation parameters
NUM_SHAPES_PER_IMAGE = (1, 3) # Min and max number of shapes per image
SHAPE_SIZE_RANGE = (10, 40)
SHAPE_INTENSITY_RANGE = (100, 255)

# Squiggle generation parameters
NUM_SQUIGGLES_PER_IMAGE = (0, 2) # Can have no squiggles
SQUIGGLE_SEGMENTS = (5, 10)
SQUIGGLE_STEP_SIZE = (5, 15)
SQUIGGLE_LINE_WIDTH = (1, 3)
SQUIGGLE_INTENSITY_RANGE = (50, 200)

# Background intensity parameters
BACKGROUND_NOISE_SCALE = 0.05
BACKGROUND_BLUR_SIGMA = 15

# Availible processors
CPUS = cpu_count()

# --- NEW: Maximum number of objects the model will predict ---
MAX_OBJECTS_PER_IMAGE = 5 # Model will always output 5*4=20 coordinates.
                          # If fewer than 5 objects, remaining bbox coords will be [0,0,0,0].

from dataset_builder import SyntheticMultiBBoxDataset

'''
# --- Helper Function for Image and Individual Bounding Box Generation ---
def generate_image_and_individual_bboxes(height, width):
    """
    Generates a single synthetic image with random shapes/squiggles and a list
    of bounding boxes, one for each generated object.

    Returns:
        tuple: A tuple containing:
            - PIL.Image: The generated image.
            - list: A list of [x_min, y_min, x_max, y_max] for each object.
    """
    # Create a base image with smoothly changing intensity
    low_res_h, low_res_w = height // 8, width // 8
    base_intensity = np.random.rand(low_res_h, low_res_w) * 255 * BACKGROUND_NOISE_SCALE
    base_intensity = gaussian_filter(base_intensity, sigma=BACKGROUND_BLUR_SIGMA / 8)
    base_intensity = np.interp(base_intensity, (base_intensity.min(), base_intensity.max()), (50, 150))
    
    base_image_pil = Image.fromarray(base_intensity.astype(np.uint8)).resize((width, height), Image.Resampling.BICUBIC)
    image_np = np.array(base_image_pil)
    image_np = np.stack([image_np, image_np, image_np], axis=-1)
    image_pil = Image.fromarray(image_np.astype(np.uint8))
    
    draw = ImageDraw.Draw(image_pil)

    individual_bboxes = [] # To store [x_min, y_min, x_max, y_max] for each object

    # --- Draw Random Shapes ---
    num_shapes = random.randint(*NUM_SHAPES_PER_IMAGE)
    for _ in range(num_shapes):
        shape_type = random.choice(['circle', 'rectangle'])
        size = random.randint(*SHAPE_SIZE_RANGE)
        
        x = random.randint(0, width - size)
        y = random.randint(0, height - size)
        
        intensity = random.randint(*SHAPE_INTENSITY_RANGE)
        shape_color = (intensity, intensity, intensity)

        bbox_coords = [x, y, x + size, y + size]

        if shape_type == 'circle':
            draw.ellipse(bbox_coords, fill=shape_color)
        else: # rectangle
            draw.rectangle(bbox_coords, fill=shape_color)
        
        individual_bboxes.append(bbox_coords)

    # --- Draw Random Squiggles ---
    num_squiggles = random.randint(*NUM_SQUIGGLES_PER_IMAGE)
    for _ in range(num_squiggles):
        num_segments = random.randint(*SQUIGGLE_SEGMENTS)
        start_x = random.randint(0, width)
        start_y = random.randint(0, height)
        points = [(start_x, start_y)]
        
        current_x, current_y = start_x, start_y
        min_x_seg, max_x_seg = current_x, current_x
        min_y_seg, max_y_seg = current_y, current_y

        for _ in range(num_segments - 1):
            step_x = random.randint(-SQUIGGLE_STEP_SIZE[1], SQUIGGLE_STEP_SIZE[1])
            step_y = random.randint(-SQUIGGLE_STEP_SIZE[1], SQUIGGLE_STEP_SIZE[1])
            
            next_x = max(0, min(width - 1, current_x + step_x))
            next_y = max(0, min(height - 1, current_y + step_y))
            
            points.append((next_x, next_y))
            current_x, current_y = next_x, next_y

            min_x_seg = min(min_x_seg, current_x)
            max_x_seg = max(max_x_seg, current_x)
            min_y_seg = min(min_y_seg, current_y)
            max_y_seg = max(max_y_seg, current_y)

        line_width = random.randint(*SQUIGGLE_LINE_WIDTH)
        squiggle_intensity = random.randint(*SQUIGGLE_INTENSITY_RANGE)
        squiggle_color = (squiggle_intensity, squiggle_intensity, squiggle_intensity)

        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i+1]
            draw.line([p1, p2], fill=squiggle_color, width=line_width)
        
        # Add squiggle bounding box (expanded by line_width)
        individual_bboxes.append([
            max(0, min_x_seg - line_width),
            max(0, min_y_seg - line_width),
            min(width, max_x_seg + line_width),
            min(height, max_y_seg + line_width)
        ])

    return image_pil, individual_bboxes

# --- PyTorch Dataset Class ---
class SyntheticMultiBBoxDataset(Dataset):
    def __init__(self, num_samples, height, width, max_objects, transforms=None):
        super().__init__()
        self.num_samples = num_samples
        self.height = height
        self.width = width
        self.max_objects = max_objects # Added max_objects
        self.transforms = transforms
        self.data = []
        
        print(f"Generating {num_samples} synthetic samples...")
        for i in range(num_samples):
            img_pil, bboxes_raw = generate_image_and_individual_bboxes(self.height, self.width)
            
            # Normalize each bounding box coordinate to [0, 1]
            bboxes_normalized = []
            for bbox_raw in bboxes_raw:
                bboxes_normalized.extend([
                    bbox_raw[0] / self.width,
                    bbox_raw[1] / self.height,
                    bbox_raw[2] / self.width,
                    bbox_raw[3] / self.height
                ])
            
            # Pad or truncate the list of bounding boxes to a fixed size
            # Each bbox has 4 coords, so target size is max_objects * 4
            target_bbox_len = self.max_objects * 4
            if len(bboxes_normalized) > target_bbox_len:
                bboxes_normalized = bboxes_normalized[:target_bbox_len]
            elif len(bboxes_normalized) < target_bbox_len:
                # Pad with zeros if fewer objects than MAX_OBJECTS_PER_IMAGE
                bboxes_normalized.extend([0.0] * (target_bbox_len - len(bboxes_normalized)))
            
            self.data.append((img_pil, bboxes_normalized))
        print("Sample generation complete.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image_pil, bboxes_normalized = self.data[idx]

        image_tensor = transforms.ToTensor()(image_pil) # Converts PIL Image to FloatTensor [0,1], C x H x W
        
        # Convert bounding box list to a float tensor
        bboxes_tensor = torch.tensor(bboxes_normalized, dtype=torch.float32)

        return image_tensor, bboxes_tensor
'''
# --- U-Net Building Blocks ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# --- U-Net Model Definition for Multi-Object Bounding Box Regression ---
class UNetMultiBBox(nn.Module):
    """
    Modified U-Net to predict a fixed number of bounding boxes for multiple objects.
    """
    def __init__(self, n_channels, n_bbox_coords_per_object=4, max_objects=MAX_OBJECTS_PER_IMAGE, bilinear=True):
        super(UNetMultiBBox, self).__init__()
        self.n_channels = n_channels
        self.total_bbox_coords = n_bbox_coords_per_object * max_objects
        self.bilinear = bilinear

        # Encoder (Downsampling Path) - Same as previous U-Net
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # Decoder (Upsampling Path) - Same as previous U-Net
        self.up1 = Up(1024 + 512, 512, bilinear)
        self.up2 = Up(512 + 256, 256, bilinear)
        self.up3 = Up(256 + 128, 128, bilinear)
        self.up4 = Up(128 + 64, 64, bilinear)

        # --- Multi-Bounding Box Prediction Head ---
        # The output feature map from the U-Net's decoder (x from self.up4) has shape (N, 64, H, W).
        # We need to distill this into a fixed number of bounding box coordinates.
        # Use Adaptive Average Pooling to reduce spatial dimensions to (1,1),
        # then a Linear layer to output the concatenated bbox coordinates.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # Reduces (H, W) to (1, 1) -> (N, 64, 1, 1)
        self.fc = nn.Linear(64, self.total_bbox_coords) # Maps 64 features to MAX_OBJECTS * 4 bbox coords
        self.sigmoid = nn.Sigmoid() # To ensure predicted coordinates are between 0 and 1

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1) # Output here: (N, 64, H, W)

        # Flatten the feature map into a vector for the linear layer
        x = self.avgpool(x) # Output: (N, 64, 1, 1)
        x = x.view(x.size(0), -1) # Flatten to (N, 64)

        bbox_preds = self.fc(x) # Output: (N, MAX_OBJECTS * 4)
        bbox_preds = self.sigmoid(bbox_preds) # Scale to [0, 1]

        return bbox_preds

# --- Create Dataset ---
print(f"Checking for existing dataset at {SAVE_PATH}...")
if os.path.exists(SAVE_PATH):
    try:
        raise SyntaxError
        # Pass max_objects during load if it's stored in the dataset object's __init__
        # For pre-generated data, it's just loading the state.
        dataset = torch.load(SAVE_PATH)
        print("Dataset loaded successfully.")
        # Ensure the loaded dataset has the same max_objects as current config, or adjust
        if hasattr(dataset, 'max_objects') and dataset.max_objects != MAX_OBJECTS_PER_IMAGE:
            print(f"Warning: Loaded dataset has max_objects={dataset.max_objects}, but config is {MAX_OBJECTS_PER_IMAGE}. Using loaded dataset's value.")
            MAX_OBJECTS_PER_IMAGE = dataset.max_objects # Adjust config to loaded dataset
    except Exception as e:
        print(f"Error loading dataset: {e}. Generating new dataset.")
        dataset = SyntheticMultiBBoxDataset(
            num_samples=NUM_SAMPLES,
            height=IMAGE_HEIGHT,
            width=IMAGE_WIDTH,
            max_objects=MAX_OBJECTS_PER_IMAGE, # Pass max_objects to dataset
            transforms=None
        )
        print(f"Saving newly generated dataset to {SAVE_PATH}...")
        torch.save(dataset, SAVE_PATH)
        print("Dataset saved successfully.")
else:
    print("No existing dataset found. Generating new dataset.")
    dataset = SyntheticMultiBBoxDataset(
        num_samples=NUM_SAMPLES,
        height=IMAGE_HEIGHT,
        width=IMAGE_WIDTH,
        max_objects=MAX_OBJECTS_PER_IMAGE, # Pass max_objects to dataset
        transforms=None
    )
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
BATCH_SIZE = 4
train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=CPUS
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=CPUS
)

# Define model parameters
INPUT_CHANNELS = 3 # RGB images
# NUM_BBOX_COORDS will now be derived from MAX_OBJECTS_PER_IMAGE * 4

# Instantiate the U-Net model for Multi-Object Bounding Box Regression
model = UNetMultiBBox(n_channels=INPUT_CHANNELS, max_objects=MAX_OBJECTS_PER_IMAGE)

print("\nU-Net Multi-Object Bounding Box Regression model created successfully!")
print(model)

# Move model to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"\nModel moved to: {device}")

print('Operating on %d cores'%CPUS)
# Define Loss Function for Regression (e.g., MSELoss)
criterion = nn.MSELoss() # Mean Squared Error Loss

# Define Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- Training Loop ---
NUM_EPOCHS = 1 # Increased epochs for multi-object regression

print(f"\n--- Starting Training for {NUM_EPOCHS} Epochs ---")

train_losses = []
val_losses = []

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for batch_idx, (images, true_bboxes) in enumerate(train_dataloader):
        images = images.to(device)
        true_bboxes = true_bboxes.to(device) # Shape: (N, MAX_OBJECTS * 4)

        optimizer.zero_grad()
        
        predicted_bboxes = model(images) # Shape: (N, MAX_OBJECTS * 4)
        
        loss = criterion(predicted_bboxes, true_bboxes)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_train_loss = running_loss / len(train_dataloader.dataset)
    train_losses.append(epoch_train_loss)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {epoch_train_loss:.6f}")

    # --- Validation Loop ---
    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for images, true_bboxes in val_dataloader:
            images = images.to(device)
            true_bboxes = true_bboxes.to(device)

            predicted_bboxes = model(images)
            loss = criterion(predicted_bboxes, true_bboxes)
            val_running_loss += loss.item() * images.size(0)

    epoch_val_loss = val_running_loss / len(val_dataloader.dataset)
    val_losses.append(epoch_val_loss)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Validation Loss: {epoch_val_loss:.6f}")

print("\n--- Training Complete ---")

# --- Visualize Training and Validation Loss ---
plt.figure(figsize=(10, 5))
plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label='Training Loss')
plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training and Validation Loss Over Epochs (Multi-Object Bounding Box Regression)')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('Training_Loss.png')
# --- Helper for drawing bounding boxes ---
def draw_bbox(ax, bbox, color='red', label=None, width=2, image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT):
    """
    Draws a single bounding box on a matplotlib axis.
    bbox is assumed to be normalized [x_min, y_min, x_max, y_max].
    """
    x_min, y_min, x_max, y_max = bbox
    x_min *= image_width
    y_min *= image_height
    x_max *= image_width
    y_max *= image_height
    
    # Check if it's a "dummy" bbox (all zeros) and skip drawing
    if x_max - x_min < 1 and y_max - y_min < 1: # Very small box (likely a padded zero)
        return

    rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                         fill=False, edgecolor=color, linewidth=width, label=label)
    ax.add_patch(rect)
    if label:
        ax.text(x_min, y_min - 5, label, color=color, fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))


# --- Example Prediction on a Validation Sample ---
print("\n--- Example Prediction on a Validation Sample ---")
model.eval() # Set model to evaluation mode
with torch.no_grad():
    # Get one sample from the validation dataset
    sample_image, true_bboxes_flat = val_dataset[0] 
    
    # Add batch dimension and move to device
    sample_image_input = sample_image.unsqueeze(0).to(device)
    
    # Get model prediction (flat tensor)
    predicted_bboxes_flat = model(sample_image_input).squeeze(0) # Remove batch dim

    # Reshape flat bbox tensors into lists of individual bboxes
    true_bboxes_list = [true_bboxes_flat[i:i+4].cpu().numpy() for i in range(0, len(true_bboxes_flat), 4)]
    predicted_bboxes_list = [predicted_bboxes_flat[i:i+4].cpu().numpy() for i in range(0, len(predicted_bboxes_flat), 4)]

    # Detach from GPU and convert to numpy for visualization
    sample_image_np = sample_image.cpu().permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 1, figsize=(8, 8))
    axes.imshow(sample_image_np)
    axes.set_title(f'Input Image with True and Predicted Bounding Boxes ({len(true_bboxes_list)} potential objects)')
    axes.axis('off')

    # Draw all true bounding boxes
    for i, bbox in enumerate(true_bboxes_list):
        draw_bbox(axes, bbox, color='green', label=f'True {i+1}' if i == 0 else None)
    
    # Draw all predicted bounding boxes
    for i, bbox in enumerate(predicted_bboxes_list):
        draw_bbox(axes, bbox, color='red', label=f'Pred {i+1}' if i == 0 else None)

    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig('output.png')

print("\nScript execution complete.")
