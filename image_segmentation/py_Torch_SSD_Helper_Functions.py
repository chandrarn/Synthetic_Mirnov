#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 13:32:01 2025
 Pytorch helper functions
@author: rianc
"""

# Libraries
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# NEW: Import SSDLite320_MobileNet_V3_Large model
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
import torch
import random

from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt
from matplotlib import rc
plt.rcParams['figure.figsize']=(6,6)
plt.rcParams['font.weight']='bold'
plt.rcParams['axes.labelweight']='bold'
plt.rcParams['lines.linewidth']=2
plt.rcParams['lines.markeredgewidth']=2
rc('font',**{'family':'serif','serif':['Palatino']})
rc('font',**{'size':11})
rc('text', usetex=True)

import os
import numpy as np

# --- Configuration Parameters ---
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
NUM_SAMPLES = 2000 # Increased samples for better training
SAVE_PATH = 'synthetic_multi_bbox_dataset_ssd.pth' # New save path for dataset
MODEL_SAVE_PATH = 'ssd_mobilenet_v3_model.pth' # Path to save the trained SSD model

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


CPUS = 28
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


MODEL_SAVE_PATH = 'ssd_mobilenet_v3_model.pth' # Path to save the trained SSD model
SAVE_PATH = 'synthetic_multi_bbox_dataset_ssd.pth' # New save path for dataset
plt.close('all')
#####################


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

# R
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

########################3
# Required to initialize dataset
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


# REquired for dataloader 
# --- Custom Collate Function for DataLoader ---
# Object detection models expect a list of (image, target) tuples in a batch.
# It handles batching internally.
def collate_fn(batch):
    return tuple(zip(*batch))


###############################################################
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

###########################################################
# --- Helper for drawing bounding boxes ---
def draw_bbox(ax, bbox, color='red', label=None, score=None, width=2,
              image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT,map_x=None, map_y=None):
    """
    Draws a single bounding box on a matplotlib axis.
    bbox is assumed to be raw [x_min, y_min, x_max, y_max] pixels.
    """
    x_min, y_min, x_max, y_max = bbox
    if map_x is not None:
        x_min, x_max = [map_x[np.floor(x).astype(int)] for x in [x_min, x_max]]
        y_min, y_max = [map_y[np.floor(y).astype(int)] for y in [y_min, y_max]]
    # Create a Rectangle patch
    rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                         fill=False, edgecolor=color, linewidth=width)
    ax.add_patch(rect)
    
    display_label = ""
    if label is not None:
        display_label += f"{label}"
    if score is not None and label is not None:
        display_label += f" | Score: {score:.2f}"
    
    shift_txt = -30 if map_x is not None else 5
    if y_min + shift_txt <0: 
        shift_txt = y_max - shift_txt
    else:   shift_txt += y_min 
    
    if map_x is not None:
        shift_txt_x = x_min if x_min + .15 < map_x[-1] else x_min - (map_x[-1] - x_min - .15 ) 
    else:shift_txt_x = x_min if x_min + 60 < 128 else x_min - (128 - x_min ) - 30 
    
    if color =='green': shift_txt_x -= 11;shift_txt-=5
    if display_label:
        ax.text(shift_txt_x, shift_txt, display_label, color=color, fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
