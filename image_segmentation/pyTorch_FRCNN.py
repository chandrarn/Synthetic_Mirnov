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

# NEW: Import Faster R-CNN model
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# CPUS
from multiprocessing import cpu_count
CPUS = cpu_count()
# --- Configuration Parameters ---
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
NUM_SAMPLES = 10 # Increased samples for better Faster R-CNN training
SAVE_PATH = 'synthetic_multi_bbox_dataset_frcnn.pth' # New save path for dataset
MODEL_SAVE_PATH = 'faster_rcnn_model.pth' # Path to save the trained Faster R-CNN model

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

# --- NEW: Number of object classes (1 for shapes/squiggles + 1 for background) ---
NUM_CLASSES = 2 # Background (0), Shape/Squiggle (1)

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

# --- PyTorch Dataset Class for Training/Validation (Adapted for Faster R-CNN) ---
class SyntheticObjectDetectionDataset(Dataset):
    def __init__(self, num_samples, height, width, transforms=None):
        super().__init__()
        self.num_samples = num_samples
        self.height = height
        self.width = width
        self.transforms = transforms
        self.data = []
        
        print(f"Generating {num_samples} synthetic samples...")
        for i in range(num_samples):
            img_pil, bboxes_raw = generate_image_and_individual_bboxes(self.height, self.width)
            
            # Convert raw bboxes to Tensor and ensure format [N, 4]
            boxes = torch.tensor(bboxes_raw, dtype=torch.float32)

            # Assign a label (1 for our single object class, 0 is background)
            labels = torch.ones((len(bboxes_raw),), dtype=torch.int64)

            # Store in Faster R-CNN target format: dictionary
            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            
            self.data.append((img_pil, target))
        print("Sample generation complete.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image_pil, target = self.data[idx]

        image_tensor = transforms.ToTensor()(image_pil) # Converts PIL Image to FloatTensor [0,1], C x H x W
        
        if self.transforms:
            # Apply transforms if any (e.g., data augmentation).
            # Note: For object detection, transforms like random flip need to update bboxes too.
            # For simplicity, we just use ToTensor here.
            pass

        return image_tensor, target

# --- NEW: Custom Collate Function for DataLoader ---
# Faster R-CNN expects a list of (image, target) tuples in a batch.
# It handles batching internally.
def collate_fn(batch):
    return tuple(zip(*batch))


# --- NEW: Dataset for Inference (no ground truth bounding boxes) ---
class InferenceDataset(Dataset):
    def __init__(self, image_dir, height, width):
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.height = height
        self.width = width
        self.transform = transforms.Compose([
            transforms.Resize((height, width)), # Ensure image size matches model input
            transforms.ToTensor()                # Convert to tensor and normalize to [0,1]
        ])
        print(f"Found {len(self.image_paths)} images for inference in {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB') # Ensure RGB format
        image_tensor = self.transform(image)
        return image_tensor, img_path # Return path for identification if needed


# --- NEW: Instantiate Faster R-CNN Model ---
def get_faster_rcnn_model(num_classes, score_thresh=0.05):
    # Load a pre-trained Faster R-CNN model with a ResNet50 FPN backbone
    # weights="DEFAULT" loads the best available weights (pretrained on COCO)
    model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT",box_score_thresh=score_thresh)

    # Get the number of input features for the classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one for our custom number of classes
    # num_classes includes the background class
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

# --- Create Dataset ---
# Note: Changed dataset class to SyntheticObjectDetectionDataset
print(f"Checking for existing dataset at {SAVE_PATH}...")
if os.path.exists(SAVE_PATH):
    try:
        dataset = torch.load(SAVE_PATH)
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading dataset: {e}. Generating new dataset.")
        dataset = SyntheticObjectDetectionDataset(
            num_samples=NUM_SAMPLES,
            height=IMAGE_HEIGHT,
            width=IMAGE_WIDTH,
            transforms=None
        )
        print(f"Saving newly generated dataset to {SAVE_PATH}...")
        torch.save(dataset, SAVE_PATH)
        print("Dataset saved successfully.")
else:
    print("No existing dataset found. Generating new dataset.")
    dataset = SyntheticObjectDetectionDataset(
        num_samples=NUM_SAMPLES,
        height=IMAGE_HEIGHT,
        width=IMAGE_WIDTH,
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
# Use the custom collate_fn for Faster R-CNN
BATCH_SIZE = 4
train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=CPUS,
    collate_fn=collate_fn # Important for Faster R-CNN
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=CPUS,
    collate_fn=collate_fn # Important for Faster R-CNN
)

# Define the desired visualization threshold for predictions (used during model initialization)
VIS_SCORE_THRESHOLD = 0.01 # TEMPORARILY LOW FOR DEBUGGING

# Instantiate the Faster R-CNN model (pass VIS_SCORE_THRESHOLD here)
model = get_faster_rcnn_model(num_classes=NUM_CLASSES, score_thresh=VIS_SCORE_THRESHOLD)

print("\nFaster R-CNN model created successfully!")
print(model)

# Move model to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"\nModel moved to: {device}")

# Optimizer
# Note: Faster R-CNN has its own internal loss calculations.
# We optimize based on the sum of these losses.
optimizer = torch.optim.Adam(model.parameters(), lr=0.005) # Increased learning rate for Faster R-CNN

# --- Training Loop ---
NUM_EPOCHS = 1 # Adjusted epochs for Faster R-CNN

print(f"\n--- Starting Training for {NUM_EPOCHS} Epochs ---")

train_losses = []
val_losses = []

for epoch in range(NUM_EPOCHS):
    model.train() # Set model to training mode
    running_loss = 0.0
    for batch_idx, (images, targets) in enumerate(train_dataloader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        
        # Faster R-CNN returns a dictionary of losses during training
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Check for non-finite loss (NaN, Inf) and skip if found
        if not torch.isfinite(losses):
            print(f"Warning: Non-finite loss encountered in epoch {epoch+1}, batch {batch_idx}. Skipping backward pass.")
            continue

        losses.backward()
        optimizer.step()

        running_loss += losses.item() * len(images) # Accumulate loss based on number of images in batch

    epoch_train_loss = running_loss / len(train_dataloader.dataset)
    train_losses.append(epoch_train_loss)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {epoch_train_loss:.6f}")

    # --- Validation Loop (FIXED) ---
    # To get losses in validation, temporarily switch to model.train()
    # within the no_grad() context.
    model.train() # Temporarily set to train mode to get loss dict output
    val_running_loss = 0.0
    with torch.no_grad(): # Disable gradient calculations
        for images, targets in val_dataloader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # In train mode, with targets, model returns a loss_dict
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Check for non-finite loss during validation as well
            if not torch.isfinite(losses):
                print(f"Warning: Non-finite validation loss encountered in epoch {epoch+1}. Skipping batch.")
                continue

            val_running_loss += losses.item() * len(images)
            
    model.eval() # Switch back to eval mode after loss calculation
    
    epoch_val_loss = val_running_loss / len(val_dataloader.dataset)
    val_losses.append(epoch_val_loss)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Validation Loss: {epoch_val_loss:.6f}")

print("\n--- Training Complete ---")

# --- Save the trained model ---
print(f"\nSaving trained model to {MODEL_SAVE_PATH}...")
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print("Model state dictionary saved successfully.")

# --- Visualize Training and Validation Loss ---
plt.figure(figsize=(10, 5))
plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label='Training Loss')
plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs (Faster R-CNN)')
plt.legend()
plt.grid(True)
plt.show()

# --- Helper for drawing bounding boxes ---
def draw_bbox(ax, bbox, color='red', label=None, score=None, width=2, image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT):
    """
    Draws a single bounding box on a matplotlib axis.
    bbox is assumed to be raw [x_min, y_min, x_max, y_max] pixels.
    """
    x_min, y_min, x_max, y_max = bbox
    
    # Create a Rectangle patch
    rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                         fill=False, edgecolor=color, linewidth=width)
    ax.add_patch(rect)
    
    display_label = ""
    if label is not None:
        display_label += f"Class: {label}"
    if score is not None:
        display_label += f" | Score: {score:.2f}"
    
    if display_label:
        ax.text(x_min, y_min - 5, display_label, color=color, fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))


# --- Example Prediction on a Validation Sample ---
print("\n--- Example Prediction on a Validation Sample (from Training Data) ---")
# Lowering SCORE_THRESHOLD significantly for initial visualization to see any boxes
# Adjust this back to a higher value (e.g., 0.5) after more training.


model.eval() # Set model to evaluation mode
with torch.no_grad():
    # Get one sample from the validation dataset
    sample_image_tensor, sample_target = val_dataset[0] 
    
    # Add batch dimension and move to device
    # Faster R-CNN expects a list of tensors for batch processing
    sample_image_input = [sample_image_tensor.to(device)]
    
    # Get model prediction (list of dicts)
    # Pass score_thresh directly for this inference call to see more detections
    predictions = model(sample_image_input,) 
    
    # --- DEBUGGING: Print raw scores ---
    if predictions and 'scores' in predictions[0]:
        print(f"\nRaw predicted scores for the validation sample: {predictions[0]['scores'].cpu().numpy()}")
    else:
        print("No predictions or scores found for the validation sample (all below internal threshold or no detections).")

    # Extract predicted boxes and true boxes for the first image in batch
    # Guard against empty predictions list
    predicted_boxes = predictions[0]['boxes'].cpu().numpy() if predictions else np.array([])
    predicted_labels = predictions[0]['labels'].cpu().numpy() if predictions else np.array([])
    predicted_scores = predictions[0]['scores'].cpu().numpy() if predictions else np.array([])

    true_boxes = sample_target['boxes'].cpu().numpy()
    true_labels = sample_target['labels'].cpu().numpy()

    # Denormalize image for display
    sample_image_np = sample_image_tensor.cpu().permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 1, figsize=(8, 8))
    axes.imshow(sample_image_np)
    axes.set_title(f'Validation Sample: Input Image with True and Predicted BBoxes (Threshold: {VIS_SCORE_THRESHOLD})')
    axes.axis('off')

    # Draw all true bounding boxes
    for i, bbox in enumerate(true_boxes):
        # Ensure bbox coordinates are in pixel space (Faster R-CNN works with pixels directly)
        draw_bbox(axes, bbox, color='green', label=f'True' if i == 0 else None, width=2)
    
    # Draw all predicted bounding boxes (filter by score)
    for i, bbox in enumerate(predicted_boxes):
        score = predicted_scores[i]
        label = predicted_labels[i]
        # No need to filter by VIS_SCORE_THRESHOLD here, as it's already applied in model(...)
        draw_bbox(axes, bbox, color='red', label=f'Pred', score=score, width=2)

    plt.legend()
    plt.tight_layout()
    plt.show()


# --- NEW SECTION: Load and Test with New Images ---
print("\n--- Testing with New, Unseen Images ---")

# 1. Create a dummy directory for new inference images if it doesn't exist
INFERENCE_IMAGE_DIR = 'new_inference_images'
os.makedirs(INFERENCE_IMAGE_DIR, exist_ok=True)

# 2. Generate a few dummy images directly into the inference directory
#    These images will NOT have corresponding ground truth bboxes.
num_inference_images_to_generate = 5
print(f"Generating {num_inference_images_to_generate} dummy images for inference in '{INFERENCE_IMAGE_DIR}'...")
for i in range(num_inference_images_to_generate):
    img_pil, _ = generate_image_and_individual_bboxes(IMAGE_HEIGHT, IMAGE_WIDTH) # Bboxes not used for inference
    img_pil.save(os.path.join(INFERENCE_IMAGE_DIR, f'inference_img_{i+1}.png'))
print("Dummy inference images generated.")


# 3. Instantiate the InferenceDataset and DataLoader
# Use collate_fn even for inference if batch size > 1
inference_dataset = InferenceDataset(INFERENCE_IMAGE_DIR, IMAGE_HEIGHT, IMAGE_WIDTH)
inference_dataloader = DataLoader(
    inference_dataset,
    batch_size=1, # Typically batch size of 1 for inference
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn # Use collate_fn
)

# 4. Load the Trained Model
print(f"\nLoading trained model from {MODEL_SAVE_PATH} for inference...")
loaded_model = get_faster_rcnn_model(num_classes=NUM_CLASSES) # Instantiate with correct num_classes
loaded_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
loaded_model.to(device)
loaded_model.eval() # Set to evaluation mode
print("Model loaded successfully for inference.")


# 5. Run Inference and Visualize Predictions
print("\nRunning inference on new images...")
for i, (image_tensors, img_paths) in enumerate(inference_dataloader): # image_tensors is a list because of collate_fn
    print(f"Processing image: {os.path.basename(img_paths[0])}")
    
    # Faster R-CNN expects a list of image tensors
    image_tensors_on_device = [img.to(device) for img in image_tensors]
    
    with torch.no_grad():
        # Pass score_thresh directly for this inference call to see more detections
        predictions = loaded_model(image_tensors_on_device) 
    
    # --- DEBUGGING: Print raw scores for inference images ---
    if predictions and 'scores' in predictions[0]:
        print(f"Raw predicted scores for inference image: {predictions[0]['scores'].cpu().numpy()}")
    else:
        print("No predictions or scores found for inference image (all below internal threshold or no detections).")

    # Extract prediction for the single image in the batch
    # Guard against empty predictions list
    predicted_boxes = predictions[0]['boxes'].cpu().numpy() if predictions else np.array([])
    predicted_labels = predictions[0]['labels'].cpu().numpy() if predictions else np.array([])
    predicted_scores = predictions[0]['scores'].cpu().numpy() if predictions else np.array([])


    # Denormalize image for display
    image_np = image_tensors[0].cpu().permute(1, 2, 0).numpy() # Take the first image from the list

    fig, axes = plt.subplots(1, 1, figsize=(8, 8))
    axes.imshow(image_np)
    axes.set_title(f'Inference: {os.path.basename(img_paths[0])} with Predicted BBoxes (Threshold: {VIS_SCORE_THRESHOLD})')
    axes.axis('off')

    # Draw all predicted bounding boxes (filter by score)
    # No need to filter by VIS_SCORE_THRESHOLD here, as it's already applied in model(...)
    for k, bbox in enumerate(predicted_boxes):
        score = predicted_scores[k]
        label = predicted_labels[k]
        draw_bbox(axes, bbox, color='red', label=f'Class {label}', score=score, width=2) # Display class label
            
    plt.legend()
    plt.tight_layout()
    plt.show()

print("\nInference on new images complete.")
print(f"You can find the generated inference images in the '{INFERENCE_IMAGE_DIR}' directory.")
