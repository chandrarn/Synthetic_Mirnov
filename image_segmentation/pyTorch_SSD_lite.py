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

# Dataset builder
from dataset_builder import generate_image_and_individual_bboxes

# --- IMPORTS for Distributed Training ---
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


# NEW: Import SSDLite320_MobileNet_V3_Large model
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
# No need to import FastRCNNPredictor anymore

# Timer
import time


from multiprocessing import cpu_count
CPUS = 4#cpu_count()

# --- Configuration Parameters ---150 
save_Ext = '_improved_model'
IMAGE_HEIGHT = 250
IMAGE_WIDTH = 250
NUM_SAMPLES = 2500 # Increased samples for better training
SAVE_PATH = 'synthetic_multi_bbox_dataset_ssd%s.pth'%save_Ext # New save path for dataset
MODEL_SAVE_PATH = 'ssd_mobilenet_v3_model%s.pth'%save_Ext # Path to save the trained SSD model

# Shape generation parameters
NUM_SHAPES_PER_IMAGE = (1, 3) # Min and max number of shapes per image
SHAPE_SIZE_RANGE = (1,7)
SHAPE_INTENSITY_RANGE = (120, 250)

# Squiggle generation parameters
NUM_SQUIGGLES_PER_IMAGE = (1, 3) # Can have no squiggles
SQUIGGLE_SEGMENTS = (5, 45)
SQUIGGLE_STEP_SIZE = (1, 5)
SQUIGGLE_LINE_WIDTH = (1, 2)
SQUIGGLE_INTENSITY_RANGE = (120,255)

# Confounding lines per image
NUM_CONFOUND_LINES = (0,3)
CONFOUND_LINE_WIDTH = (1,2)
CONFOUND_LINE_INTENSITY_RANGE = (50,120)

# Background intensity parameters
BACKGROUND_NOISE_SCALE = 0.05
BACKGROUND_BLUR_SIGMA = 15

# --- NEW: Number of object classes (1 for shapes/squiggles + 1 for background) ---
NUM_CLASSES = 2 # Background (0), Shape/Squiggle (1)

NUM_EPOCHS = 190 # Adjusted epochs for SSD
# Define the desired visualization threshold for predictions (used during model initialization)
VIS_SCORE_THRESHOLD = 0.7 # TEMPORARILY LOW FOR DEBUGGING
plt.close('all')
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
        
        # Allow intensity, width to change within some small range
        start_line_width = random.randint(*SQUIGGLE_LINE_WIDTH)
        start_squiggle_intensity = random.randint(*SQUIGGLE_INTENSITY_RANGE)
        
        widths = [start_line_width]
        intensities = [start_squiggle_intensity]

        for _ in range(num_segments - 1):
            step_x = random.randint(1, SQUIGGLE_STEP_SIZE[1])
            step_y = random.randint(-SQUIGGLE_STEP_SIZE[1], SQUIGGLE_STEP_SIZE[1])
            
            next_x = max(0, min(width - 1, current_x + step_x))
            next_y = max(0, min(height - 1, current_y + step_y))
            
            points.append((next_x, next_y))
            current_x, current_y = next_x, next_y

            min_x_seg = min(min_x_seg, current_x)
            max_x_seg = max(max_x_seg, current_x)
            min_y_seg = min(min_y_seg, current_y)
            max_y_seg = max(max_y_seg, current_y)
            
            # Change widths, amplitudes
            # Step width
            step_width = random.randint(-1,1)
            if not (SQUIGGLE_LINE_WIDTH[0] <= (widths[-1] + \
                                       step_width) <= SQUIGGLE_LINE_WIDTH[1]):
                step_width = 0
            widths.append(widths[-1] + step_width)
            
            # step intensity
            step_intensity = random.randint(-20,20)
            if not (SQUIGGLE_INTENSITY_RANGE[0] <= (intensities[-1] + \
                                step_intensity) <= SQUIGGLE_INTENSITY_RANGE[1]):
                step_intensity = 0
            intensities.append(intensities[-1] + step_intensity)

        #line_width = random.randint(*SQUIGGLE_LINE_WIDTH)
        #squiggle_intensity = random.randint(*SQUIGGLE_INTENSITY_RANGE)
        #squiggle_color = (squiggle_intensity, squiggle_intensity, squiggle_intensity)

        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i+1]
            intensity = (intensities[i],intensities[i],intensities[i])
            draw.line([p1, p2], fill=intensity, width=widths[i])
        
        # Add squiggle bounding box (expanded by line_width)
        individual_bboxes.append([
            max(0, min_x_seg - max(widths)),
            max(0, min_y_seg - max(widths)),
            min(width, max_x_seg + max(widths)),
            min(height, max_y_seg + max(widths))
        ])
        
    num_confounding_lines = random.randint(*NUM_CONFOUND_LINES)
    
    for _ in range(num_confounding_lines):
        # Randomize width
        line_width = random.randint(*CONFOUND_LINE_WIDTH)
        # randomize color
        confound_line_intensity = random.randint(*(CONFOUND_LINE_INTENSITY_RANGE))
        confound_color = (confound_line_intensity,confound_line_intensity,confound_line_intensity)
        if random.randint(0,1)==0: # start on x axis
            start_x = random.randint(0, width)
            draw.line([(start_x,0),(start_x,height)], fill=confound_color, width = line_width)
        else: 
            start_y = random.randint(0, height)
            draw.line([(0, start_y),(width,start_y)], fill=confound_color, width = line_width)

    return image_pil, individual_bboxes
'''
import math
# --- Configuration Parameters (can be adjusted) ---
IMAGE_HEIGHT = 250
IMAGE_WIDTH = 250

# Circle chain parameters
ADD_CIRCLE_CHAIN_PROBABILITY = 0.7 # Probability of adding a circle chain (0.0 to 1.0)
NUM_CIRCLES_IN_CHAIN = (10, 30) # Min and max number of circles in a horizontal chain
CIRCLE_SPACING_FACTOR = 1. # Factor of circle radius for spacing between centers
VERTICAL_PERTURBATION_STD = 2 # Standard deviation for vertical offset of circles
GAUSSIAN_SIGMA_FACTOR = 0.3 # Controls sharpness of Gaussian intensity peak (relative to radius)
CIRCLE_CHAIN_VERTICAL_RESTRICTION_FACTOR = 5/6 # Restrict to lower 1/6th of the image (starts at 5/6 * height)
CIRCLE_CHAIN_RADIUS_RANGE = (5, 9) # Radius range for circles in chain
SHAPE_INTENSITY_RANGE = (150,255)

# Rectangle noise parameters (no bounding box)
NUM_RECTANGLES_AS_NOISE = (0, 2) # Min and max number of rectangles to draw as noise
RECTANGLE_SIZE_RANGE = (10, 40)
RECTANGLE_INTENSITY_RANGE = (50, 120)

# Squiggle generation parameters (with bounding box)
NUM_SQUIGGLES_PER_IMAGE = (1, 3) # Can have no squiggles
SQUIGGLE_SEGMENTS = (10, 20)
SQUIGGLE_STEP_SIZE_HORIZONTAL = (10, 25) # Dominantly horizontal steps
SQUIGGLE_STEP_SIZE_VERTICAL = (1, 5)   # Small vertical steps
SQUIGGLE_LINE_WIDTH = (3, 7)
SQUIGGLE_INTENSITY_START_RANGE = (50, 150) # Range for start intensity
SQUIGGLE_INTENSITY_END_RANGE = (150, 255) # Range for end intensity
SQUIGGLE_VERTICAL_RESTRICTION_FACTOR = 16/18

# Straight Line noise parameters (no bounding box)
NUM_STRAIGHT_LINES_AS_NOISE = (0, 5) # Min and max number of straight lines to draw as noise
STRAIGHT_LINE_WIDTH = (2, 6)
STRAIGHT_LINE_INTENSITY_RANGE = (50, 200)


# Background intensity parameters
BACKGROUND_NOISE_SCALE = 0.05
BACKGROUND_BLUR_SIGMA = 15

# --- Helper function for Gaussian intensity circle ---
def draw_gaussian_circle(image_array, center_x, center_y, radius, peak_intensity, sigma_factor):
    """
    Draws a circle with Gaussian intensity profile on a numpy image array.
    """
    h, w = image_array.shape[:2]
    
    # Calculate bounding box for the circle to optimize pixel iteration
    min_x = max(0, int(center_x - radius))
    max_x = min(w, int(center_x + radius) + 1)
    min_y = max(0, int(center_y - radius))
    max_y = min(h, int(center_y + radius) + 1)

    sigma = radius * sigma_factor
    if sigma == 0: # Avoid division by zero for tiny circles
        sigma = 1.0

    for y in range(min_y, max_y):
        for x in range(min_x, max_x):
            dist_sq = (x - center_x)**2 + (y - center_y)**2
            if dist_sq <= radius**2: # Only draw within the circle
                # Gaussian falloff
                intensity_factor = math.exp(-dist_sq / (2 * sigma**2))
                current_intensity = image_array[y, x, 0] # Assuming grayscale for simplicity, or average
                
                # Apply new intensity, blending with existing content
                # Ensure it's within 0-255 range and increases intensity
                new_intensity_val = min(255, max(current_intensity, int(peak_intensity * intensity_factor)))
                
                image_array[y, x] = [new_intensity_val, new_intensity_val, new_intensity_val] # Apply to all channels


# --- Main Image and Bounding Box Generation Function ---
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
    image_np = np.stack([image_np, image_np, image_np], axis=-1) # Ensure 3 channels for PIL drawing
    
    # Create a mutable PIL Image from the numpy array for drawing
    image_pil = Image.fromarray(image_np.astype(np.uint8))
    draw = ImageDraw.Draw(image_pil)

    individual_bboxes = [] # To store [x_min, y_min, x_max, y_max] for each object

    # --- Draw Circle Chain (0 or 1) ---
    if random.random() < ADD_CIRCLE_CHAIN_PROBABILITY:
        num_circles = random.randint(*NUM_CIRCLES_IN_CHAIN)
        circle_radius = random.randint(*CIRCLE_CHAIN_RADIUS_RANGE)
        
        # Determine starting point for the chain, restricted to lower 1/6th
        min_y_for_chain = int(height * CIRCLE_CHAIN_VERTICAL_RESTRICTION_FACTOR)
        
        # Ensure the chain fits within the image bounds horizontally
        chain_width_estimate = num_circles * (circle_radius * 2 * CIRCLE_SPACING_FACTOR)
        start_x = random.randint(0, max(0, width - int(chain_width_estimate)))
        
        # start_y should ensure the TOP of the first circle is within the lower 1/6th
        # and the BOTTOM of the circle is within image height.
        # Center of the circle:
        start_y = random.randint(min_y_for_chain + circle_radius, height - circle_radius) 

        current_x = start_x
        current_y = start_y # Base y for the chain, perturbations apply to this

        # Initialize chain bounding box to extreme values
        chain_min_x = width
        chain_min_y = height
        chain_max_x = 0
        chain_max_y = 0

        for i in range(num_circles):
            # Apply vertical perturbation
            perturb_y = random.gauss(0, VERTICAL_PERTURBATION_STD)
            current_y_perturbed = max(min_y_for_chain + circle_radius, min(height - circle_radius, current_y + perturb_y))

            # Vary peak intensity slightly
            peak_intensity = random.randint(*SHAPE_INTENSITY_RANGE)
            
            # Draw Gaussian circle directly on the numpy array
            # Convert PIL image to numpy array for pixel manipulation
            image_np = np.array(image_pil)
            draw_gaussian_circle(image_np, current_x + circle_radius, current_y_perturbed, circle_radius, peak_intensity, GAUSSIAN_SIGMA_FACTOR)
            image_pil = Image.fromarray(image_np.astype(np.uint8)) # Convert back to PIL for next draw call
            draw = ImageDraw.Draw(image_pil) # Re-initialize Draw object

            # Update overall chain bounding box
            chain_min_x = min(chain_min_x, int(current_x))
            chain_min_y = min(chain_min_y, int(current_y_perturbed - circle_radius))
            chain_max_x = max(chain_max_x, int(current_x + 2 * circle_radius))
            chain_max_y = max(chain_max_y, int(current_y_perturbed + circle_radius))

            # Move to the next circle position
            current_x += int(circle_radius * 2 * CIRCLE_SPACING_FACTOR)
        
        # Add the single bounding box for the entire chain
        individual_bboxes.append([chain_min_x, chain_min_y, chain_max_x, chain_max_y])


    # --- Draw Random Rectangles (Noise - No Bounding Box) ---
    num_rectangles = random.randint(*NUM_RECTANGLES_AS_NOISE)
    for _ in range(num_rectangles):
        size = random.randint(*RECTANGLE_SIZE_RANGE)
        x = random.randint(0, width - size)
        y = random.randint(0, height - size)
        intensity = random.randint(*RECTANGLE_INTENSITY_RANGE)
        shape_color = (intensity, intensity, intensity)
        bbox_coords = [x, y, x + size, y + size]
        draw.rectangle(bbox_coords, fill=shape_color)
        # Bounding box is NOT added for noise rectangles

    # --- Draw Random Squiggles (With Bounding Box) ---
    num_squiggles = random.randint(*NUM_SQUIGGLES_PER_IMAGE)
    for _ in range(num_squiggles):
        num_segments = random.randint(*SQUIGGLE_SEGMENTS)
        
        # Start point for the squiggle, biased horizontally
        start_x = random.randint(0, width // 4) # Start from left quarter
        start_y = random.randint(0, int(height * SQUIGGLE_VERTICAL_RESTRICTION_FACTOR))
        points = [(start_x, start_y)]
        
        current_x, current_y = start_x, start_y
        min_x_seg, max_x_seg = current_x, current_x
        min_y_seg, max_y_seg = current_y, current_y

        # Define start and end intensity for the squiggle
        start_intensity = random.randint(*SQUIGGLE_INTENSITY_START_RANGE)
        end_intensity = random.randint(*SQUIGGLE_INTENSITY_END_RANGE)

        for i in range(num_segments - 1):
            # Dominantly horizontal step
            step_x = random.randint(*SQUIGGLE_STEP_SIZE_HORIZONTAL)
            # Small vertical perturbation
            step_y = random.randint(-SQUIGGLE_STEP_SIZE_VERTICAL[1], SQUIGGLE_STEP_SIZE_VERTICAL[1])
            
            next_x = max(0, min(width - 1, current_x + step_x))
            next_y = max(0, min(height - 1, current_y + step_y))
            
            points.append((next_x, next_y))
            
            # Interpolate intensity for the current segment
            t = (i ) / (num_segments - 1) if num_segments > 1 else 0 # Normalized position along squiggle
            segment_intensity = int(start_intensity * (1 - t) + end_intensity * t)
            squiggle_color = (segment_intensity, segment_intensity, segment_intensity)

            line_width = random.randint(*SQUIGGLE_LINE_WIDTH)
            
            # Draw the segment
            draw.line([points[i], points[i+1]], fill=squiggle_color, width=line_width)

            current_x, current_y = next_x, next_y

            min_x_seg = min(min_x_seg, current_x)
            max_x_seg = max(max_x_seg, current_x)
            min_y_seg = min(min_y_seg, current_y)
            max_y_seg = max(max_y_seg, current_y)

        # Add squiggle bounding box (expanded by line_width)
        individual_bboxes.append([
            max(0, min_x_seg - line_width),
            max(0, min_y_seg - line_width),
            min(width, max_x_seg + line_width),
            min(height, max_y_seg + line_width)
        ])

    # --- Draw Random Straight Lines (Noise - No Bounding Box) ---
    num_straight_lines = random.randint(*NUM_STRAIGHT_LINES_AS_NOISE)
    for _ in range(num_straight_lines):
        line_width = random.randint(*STRAIGHT_LINE_WIDTH)
        intensity = random.randint(*STRAIGHT_LINE_INTENSITY_RANGE)
        line_color = (intensity, intensity, intensity)

        if random.random() < 0.5: # 50% chance for horizontal or vertical
            # Horizontal line
            x1 = random.randint(0, width - 1)
            x2 = random.randint(x1, width - 1) # Ensure x2 >= x1
            y = random.randint(0, height - 1)
            draw.line([(x1, y), (x2, y)], fill=line_color, width=line_width)
        else:
            # Vertical line
            y1 = random.randint(0, height - 1)
            y2 = random.randint(y1, height - 1) # Ensure y2 >= y1
            x = random.randint(0, width - 1)
            draw.line([(x, y1), (x, y2)], fill=line_color, width=line_width)
        # Bounding box is NOT added for noise straight lines

    return image_pil, individual_bboxes

# --- PyTorch Dataset Class for Training/Validation (Adapted for Object Detection) ---
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

            # Store in Object Detection target format: dictionary
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

# --- Custom Collate Function for DataLoader ---
# Object detection models expect a list of (image, target) tuples in a batch.
# It handles batching internally.
def collate_fn(batch):
    return tuple(zip(*batch))


# --- Dataset for Inference (no ground truth bounding boxes) ---
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


# --- Instantiate SSDLite320_MobileNet_V3_Large Model ---
def get_ssd_model(num_classes, score_thresh=0.05):
    # 1. Initialize the model with our custom num_classes and score_thresh, but NO pre-trained weights initially.
    #    This ensures the head is correctly sized for our task.
    model = ssdlite320_mobilenet_v3_large(weights=None, num_classes=num_classes, score_thresh=score_thresh)

    # 2. Load the pre-trained weights *selectively* from a default model.
    #    We create a dummy model just to get its pre-trained state_dict.
    pretrained_model = ssdlite320_mobilenet_v3_large(weights=None)
    pretrained_state_dict = pretrained_model.state_dict()

    # 3. Filter out the keys corresponding to the original classification and regression heads.
    #    These heads were designed for 91 COCO classes and won't fit our 2-class setup.
    #    The `head` attribute contains `classification_head` and `regression_head` in SSDLite.
    new_state_dict = {}
    for k, v in pretrained_state_dict.items():
        if not (k.startswith("head.classification_head") or k.startswith("head.regression_head")):
            new_state_dict[k] = v

    # 4. Load the filtered state_dict into our new model.
    #    `strict=False` is important here because we expect missing keys (our new head's layers)
    #    and unexpected keys (the old head's layers that were filtered out).
    model.load_state_dict(new_state_dict, strict=False)

    return model

# --- Create Dataset ---
print(f"Checking for existing dataset at {SAVE_PATH}...")
if os.path.exists(SAVE_PATH):
    try:
        raise SyntaxError
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
BATCH_SIZE = 4
train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=CPUS,
    collate_fn=collate_fn # Important for Object Detection Models
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=CPUS,
    collate_fn=collate_fn # Important for Object Detection Models
)



# Instantiate the SSDLite model
model = get_ssd_model(num_classes=NUM_CLASSES, score_thresh=VIS_SCORE_THRESHOLD)

print("\nSSDLite320_MobileNet_V3_Large model created successfully!")
# print(model)

# Move model to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Multi-GPU support ---
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
    model = nn.DataParallel(model)
model.to(device)
print(f"\nModel moved to: {device}")

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.005) # Increased learning rate

# --- Training Loop ---

print('Running on %d cores'%CPUS)
print(f"\n--- Starting Training for {NUM_EPOCHS} Epochs ---")
start_time = time.time()


train_losses = []
val_losses = []

for epoch in range(NUM_EPOCHS):
    model.train() # Set model to training mode
    running_loss = 0.0
    for batch_idx, (images, targets) in enumerate(train_dataloader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        
        # SSDLite returns a dictionary of losses during training
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        if isinstance(losses, torch.Tensor) and losses.numel() > 1:
            losses = losses.mean()
        if not torch.isfinite(losses):
            print(f"Warning: Non-finite loss encountered in epoch {epoch+1}, batch {batch_idx}. Skipping backward pass.")
            continue

        losses.backward()
        optimizer.step()

        running_loss += losses.item() * len(images) # Accumulate loss based on number of images in batch

    epoch_train_loss = running_loss / len(train_dataloader.dataset)
    train_losses.append(epoch_train_loss)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {epoch_train_loss:.6f}, Elapsed Time: {time.time() - start_time:.2f}s")

    # --- Validation Loop ---
    # Temporarily set to train mode to get loss dict output, then switch back to eval mode.
    model.train() 
    val_running_loss = 0.0
    with torch.no_grad(): # Disable gradient calculations
        for images, targets in val_dataloader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            if isinstance(losses, torch.Tensor) and losses.numel() > 1:
                losses = losses.mean()
            if not torch.isfinite(losses):
                print(f"Warning: Non-finite loss encountered in epoch {epoch+1}, batch {batch_idx}. Skipping backward pass.")
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
plt.title('Training and Validation Loss Over Epochs (SSDLite MobileNetV3)')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('Training_Progress%.png'%save_Ext)
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

model.eval() # Set model to evaluation mode
with torch.no_grad():
    # Get one sample from the validation dataset
    sample_image_tensor, sample_target = val_dataset[0] 
    
    # Add batch dimension and move to device
    sample_image_input = [sample_image_tensor.to(device)]
    
    # Get model prediction (list of dicts)
    predictions = model(sample_image_input) 
    
    # --- DEBUGGING: Print raw scores ---
    if predictions and 'scores' in predictions[0]:
        print(f"\nRaw predicted scores for the validation sample: {predictions[0]['scores'].cpu().numpy()}")
    else:
        print("No predictions or scores found for the validation sample (all below internal threshold or no detections).")

    # Extract predicted boxes and true boxes for the first image in batch
    predicted_boxes = predictions[0]['boxes'].cpu().numpy() if predictions else np.array([])
    predicted_labels = predictions[0]['labels'].cpu().numpy() if predictions else np.array([])
    predicted_scores = predictions[0]['scores'].cpu().numpy() if predictions else np.array([])

    true_boxes = sample_target['boxes'].cpu().numpy()
    true_labels = sample_target['labels'].cpu().numpy()

    # Denormalize image for display
    sample_image_np = sample_image_tensor.cpu().permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 1, figsize=(8, 8))
    axes.imshow(sample_image_np)
    axes.set_title(f'Validation Sample: Input Image with True and Predicted BBoxes (Model Threshold: {VIS_SCORE_THRESHOLD})')
    axes.axis('off')

    # Draw all true bounding boxes
    for i, bbox in enumerate(true_boxes):
        draw_bbox(axes, bbox, color='green', label=f'True' if i == 0 else None, width=2)
    
    # Draw all predicted bounding boxes (filtered by the model's internal threshold)
    for i, bbox in enumerate(predicted_boxes):
        score = predicted_scores[i]
        label = predicted_labels[i]
        draw_bbox(axes, bbox, color='red', label=f'Pred', score=score, width=2)

    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig('Training_Validation%s.png'%save_Ext)

#raise SyntaxError
# --- NEW SECTION: Load and Test with New Images ---
print("\n--- Testing with New, Unseen Images ---")

# 1. Create a dummy directory for new inference images if it doesn't exist
INFERENCE_IMAGE_DIR = 'new_inference_images'
os.makedirs(INFERENCE_IMAGE_DIR, exist_ok=True)

# 2. Generate a few dummy images directly into the inference directory
#    These images will NOT have corresponding ground truth bboxes.
num_inference_images_to_generate = 108
print(f"Generating {num_inference_images_to_generate} dummy images for inference in '{INFERENCE_IMAGE_DIR}'...")
for i in range(num_inference_images_to_generate):
    img_pil, _ = generate_image_and_individual_bboxes(IMAGE_HEIGHT, IMAGE_WIDTH) # Bboxes not used for inference
    img_pil.save(os.path.join(INFERENCE_IMAGE_DIR, f'inference_img_{i+1}.png'))
print("Dummy inference images generated.")

INFERENCE_IMAGE_DIR = '../output_plots/training_plots/test_images/test_again/'
# 3. Instantiate the InferenceDataset and DataLoader
inference_dataset = InferenceDataset(INFERENCE_IMAGE_DIR, IMAGE_HEIGHT, IMAGE_WIDTH)
inference_dataloader = DataLoader(
    inference_dataset,
    batch_size=1, # Typically batch size of 1 for inference
    shuffle=False,
    num_workers=CPUS,
    collate_fn=collate_fn # Use collate_fn
)

# 4. Load the Trained Model
VIS_SCORE_THRESHOLD = .1
print(f"\nLoading trained model from {MODEL_SAVE_PATH} for inference...")
state_dict = torch.load(MODEL_SAVE_PATH, map_location=device)
if any(k.startswith('module.') for k in state_dict.keys()):
    # State dict was saved from DataParallel, strip 'module.' prefix
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[name] = v
    state_dict = new_state_dict

loaded_model = get_ssd_model(num_classes=NUM_CLASSES, score_thresh=VIS_SCORE_THRESHOLD)
loaded_model.load_state_dict(state_dict)

if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    loaded_model = nn.DataParallel(loaded_model)
loaded_model.to(device)
loaded_model.eval()
print("Model loaded successfully for inference.")


# 5. Run Inference and Visualize Predictions
print("\nRunning inference on new images...")
for i, (image_tensors, img_paths) in enumerate(inference_dataloader):
    print(f"Processing image: {os.path.basename(img_paths[0])}")
    if i > 5: break
    image_tensors_on_device = [img.to(device) for img in image_tensors]
    
    with torch.no_grad():
        predictions = loaded_model(image_tensors_on_device) 
    
    if predictions and 'scores' in predictions[0]:
        #print(f"Raw predicted scores for inference image: {predictions[0]['scores'].cpu().numpy()}")
        print('Score found for test image')
    else:
        print("No predictions or scores found for inference image (all below internal threshold or no detections).")

    predicted_boxes = predictions[0]['boxes'].cpu().numpy() if predictions else np.array([])
    predicted_labels = predictions[0]['labels'].cpu().numpy() if predictions else np.array([])
    predicted_scores = predictions[0]['scores'].cpu().numpy() if predictions else np.array([])


    # Denormalize image for display
    image_np = image_tensors[0].cpu().permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 1, figsize=(8, 8))
    axes.imshow(image_np)
    axes.set_title(f'Inference: {os.path.basename(img_paths[0])} with Predicted BBoxes (Model Threshold: {VIS_SCORE_THRESHOLD})')
    axes.axis('off')

    for k, bbox in enumerate(predicted_boxes):
        score = predicted_scores[k]
        label = predicted_labels[k]
        draw_bbox(axes, bbox, color='red', label=f'Class {label}', score=score, width=2)
            
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(INFERENCE_IMAGE_DIR+'Training_Prediction_%d%s.png'%(i,save_Ext))

print("\nInference on new images complete.")
print(f"You can find the generated inference images in the '{INFERENCE_IMAGE_DIR}' directory.")
