#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 14:21:46 2025
    Synthetic shape generator/model training

    - generate arbitrary number of arbitratilly placed shapes with bounding boxes
    - the shapes can have randomly generated, smoothly changing amplitudes
    - add lines, background noise 
@author: rianc
"""
    
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


# --- Configuration Parameters ---
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
# Number of image-mask pairs to generate

# Shape generation parameters
NUM_SHAPES_PER_IMAGE = (2, 5) # Min and max number of shapes per image
SHAPE_SIZE_RANGE = (2, 15) # Min and max size for shapes
SHAPE_INTENSITY_RANGE = (100, 255) # Intensity range for shapes (0-255)

# Squiggle generation parameters
NUM_SQUIGGLES_PER_IMAGE = (1, 3) # Min and max number of squiggles per image
SQUIGGLE_SEGMENTS = (5, 45) # Min and max number of segments in a squiggle
SQUIGGLE_STEP_SIZE = (1, 5) # Max step size for each segment of a squiggle
SQUIGGLE_LINE_WIDTH = (1, 5) # Line width for squiggles
SQUIGGLE_INTENSITY_RANGE = (90, 200) # Intensity range for squiggles

# Confounding lines per image
NUM_CONFOUND_LINES = (0,3)
CONFOUND_LINE_WIDTH = (1,2)
CONFOUND_LINE_INTENSITY_RANGE = (50,100)
# Background intensity parameters
BACKGROUND_NOISE_SCALE = 0.05 # How much initial random noise for background
BACKGROUND_BLUR_SIGMA = 15 # How much to blur the background noise for smooth changes



MAX_OBJECTS_PER_IMAGE = 5




# --- Helper Function for Image and Mask Generation ---
def generate_image_and_individual_bboxes(height, width):
    """
    Generates a single synthetic image with random shapes/squiggles and a corresponding
    binary mask containing bounding boxes around the generated objects.
    
    ADD: 
        Confounding straight lines

    Args:
        height (int): The height of the output image.
        width (int): The width of the output image.

    Returns:
        tuple: A tuple containing:
            - PIL.Image: The generated image.
            - PIL.Image: The binary mask (0 for background, 255 for foreground).
    """

    # Create a base image with smoothly changing intensity
    # Generate low-res random noise and blur, then resize
    low_res_h, low_res_w = height // 8, width // 8
    base_intensity = np.random.rand(low_res_h, low_res_w) * 255 * BACKGROUND_NOISE_SCALE
    base_intensity = gaussian_filter(base_intensity, sigma=BACKGROUND_BLUR_SIGMA / 8) # Blur low-res
    base_intensity = np.interp(base_intensity, (base_intensity.min(), base_intensity.max()), (50, 150)) # Scale to a visible range
    
    # Resize back to full resolution using PIL for interpolation
    base_image_pil = Image.fromarray(base_intensity.astype(np.uint8)).resize((width, height), Image.Resampling.BICUBIC)
    image = np.array(base_image_pil)
    image = np.stack([image, image, image], axis=-1) # Convert to RGB
    image = Image.fromarray(image.astype(np.uint8))
    
    draw = ImageDraw.Draw(image)

    # Create an empty mask
    mask = Image.new('L', (width, height), 0) # 'L' for grayscale, 0 for black background
    mask_draw = ImageDraw.Draw(mask)

    all_bounding_boxes = []

    # --- Draw Random Shapes ---
    num_shapes = random.randint(*NUM_SHAPES_PER_IMAGE)
    for _ in range(num_shapes):
        shape_type = random.choice(['circle', 'rectangle'])
        size = random.randint(*SHAPE_SIZE_RANGE)
        
        # Ensure shape fits within image boundaries
        x = random.randint(0, width - size)
        y = random.randint(0, height - size)
        
        intensity = random.randint(*SHAPE_INTENSITY_RANGE)
        shape_color = (intensity, intensity, intensity) # Grayscale shapes

        if shape_type == 'circle':
            bbox = [x, y, x + size, y + size]
            draw.ellipse(bbox, fill=shape_color)
        else: # rectangle
            bbox = [x, y, x + size, y + size]
            draw.rectangle(bbox, fill=shape_color)
        
        all_bounding_boxes.append(bbox)

    # --- Draw Random Squiggles ---
    num_squiggles = random.randint(*NUM_SQUIGGLES_PER_IMAGE)
    for _ in range(num_squiggles):
        num_segments = random.randint(*SQUIGGLE_SEGMENTS)
        start_x = random.randint(0, width)
        start_y = random.randint(0, height)
        points = [(start_x, start_y)]
        
        current_x, current_y = start_x, start_y
        min_x, max_x = current_x, current_x
        min_y, max_y = current_y, current_y

        # Allow intensity, width to change within some small range
        start_line_width = random.randint(*SQUIGGLE_LINE_WIDTH)
        start_squiggle_intensity = random.randint(*SQUIGGLE_INTENSITY_RANGE)
        
        widths = [start_line_width]
        intensities = [start_squiggle_intensity]
        
        
        #squiggle_color = (squiggle_intensity, squiggle_intensity, squiggle_intensity)
        for _ in range(num_segments - 1):
            step_x = random.randint(1, SQUIGGLE_STEP_SIZE[1])
            step_y = random.randint(-SQUIGGLE_STEP_SIZE[1], SQUIGGLE_STEP_SIZE[1])
            
            # Ensure points stay within bounds
            next_x = max(0, min(width - 1, current_x + step_x))
            next_y = max(0, min(height - 1, current_y + step_y))
            
            points.append((next_x, next_y))
            current_x, current_y = next_x, next_y

            min_x = min(min_x, current_x)
            max_x = max(max_x, current_x)
            min_y = min(min_y, current_y)
            max_y = max(max_y, current_y)
            
            widths.append(start_line_width)
            intensities.append(start_squiggle_intensity)
            '''
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
            '''
        # Draw line segment by segment to apply smooth intensity changes
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i+1]
            intensity = (intensities[i],intensities[i],intensities[i])
            draw.line([p1, p2], fill=intensity, width=widths[i])
        
        # Calculate bounding box for the entire squiggle (including line width)
        # bbox_squiggle = [
        #     max(0, min_x - max(widths)),
        #     max(0, min_y - max(widths)),
        #     min(width, max_x + max(widths)),
        #     min(height, max_y + max(widths))
        # ]
        bbox_squiggle = [
            min_x,min_y,max_x,max_y
            ]
        all_bounding_boxes.append(bbox_squiggle)

    # --- Add straight lines as confounding variables
    
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
    
    
    '''
    # --- Create Mask from Bounding Boxes ---
    for bbox in all_bounding_boxes:
        # Fill the bounding box in the mask with white (255)
        mask_draw.rectangle(bbox, fill=None,outline='white')
    '''
    return image, all_bounding_boxes

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

# --- Visualize Generated Samples ---
def visualize_sample(image_tensor, mask_tensor, title=""):
    """
    Visualizes an image and its corresponding mask.
    Args:
        image_tensor (torch.Tensor): Image tensor (C, H, W)
        mask_tensor (torch.Tensor): Mask tensor (C, H, W or H, W)
    """
    plt.close(title)
    fig, axes = plt.subplots(1, 1, figsize=(10, 5),num=title)
    fig.suptitle(title)

    # Convert image tensor back to numpy and adjust channels for matplotlib (H, W, C)
    img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    axes.imshow(img_np)
    axes.set_title('Generated Image')
    axes.axis('off')

    # Convert mask tensor back to numpy (remove channel dimension if present)
    # mask_np = mask_tensor.squeeze().cpu().numpy()
    # axes[1].imshow(mask_np, cmap='gray') # Use grayscale colormap for masks
    # axes[1].set_title('Bounding Box Mask')
    # axes[1].axis('off')
    draw_bbox(axes, mask_tensor)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

'''
def build_PyTorch_Loader(doSave=True,NUM_SAMPLES = 10,doPlot=True):
    # Get and visualize a few batches
    
    # --- Data Transformation Pipeline ---
    # For this example, we'll only convert to tensor.
    # More advanced transforms (e.g., augmentation) should be applied carefully
    # to both image and mask simultaneously.
    # The Dataset class handles the ToTensor conversion for image and mask.

    # --- Create Dataset and DataLoader ---
    dataset = SyntheticMultiBBoxDataset(
        num_samples=NUM_SAMPLES,
        height=IMAGE_HEIGHT,
        width=IMAGE_WIDTH,
        max_objects=MAX_OBJECTS_PER_IMAGE, # Pass max_objects to dataset
        transforms=True # Transforms are applied inside __getitem__ for simplicity
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4, # Example batch size
        shuffle=True,
        num_workers=0 # Set to >0 for multiprocessing data loading in a real scenario
    )


    
    if doPlot:
        print(f"Generating {NUM_SAMPLES} samples. Displaying first few batches.")
        for i, (images, masks) in enumerate(dataloader):
            print(f"Batch {i+1}: Images shape: {images.shape}, Masks shape: {masks.shape}")
            return images,masks
            # Visualize samples from the current batch
            for j in range(images.shape[0]):
                visualize_sample(images[j], masks[j], title=f"Sample from Batch {i+1}, Item {j+1}")
                if j >= 1: # Only show first 2 per batch for brevity
                    break
            if i >= 1: # Only show first 2 batches for brevity
                break
    
    print("\nDataset generation and visualization complete.")
    
    if doSave: 
        # Save the dataset
        SAVE_PATH = 'dataset.pt'
        print(f"Saving dataset to {SAVE_PATH}...")
        try:
            torch.save(dataset, SAVE_PATH)
            print("Dataset saved successfully.")
        except Exception as e:
            print(f"Error saving dataset: {e}")
'''