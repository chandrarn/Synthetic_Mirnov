#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 16:20:37 2025
  Validadtion on pre-trained model
@author: rianc
"""

# Libraries
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# NEW: Import SSDLite320_MobileNet_V3_Large model
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
import torch

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

import os
import numpy as np

from py_Torch_SSD_Helper_Functions import InferenceDataset, collate_fn,\
    get_ssd_model, draw_bbox, SyntheticObjectDetectionDataset


IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128


CPUS = 28
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Number of object classes (1 for shapes/squiggles + 1 for background)
NUM_CLASSES = 2 # Background (0), Shape/Squiggle (1)

MODEL_SAVE_PATH = 'ssd_mobilenet_v3_model.pth' # Path to save the trained SSD model
SAVE_PATH = 'synthetic_multi_bbox_dataset_ssd.pth' # New save path for dataset
plt.close('all')



#############################################################################
###################3333
INFERENCE_IMAGE_DIR = '../output_plots/training_plots/test_images/test_again/'
INFERENCE_IMAGE_DIR_SAVE = '../output_plots/training_plots/test_images/test_again/out/'
# 3. Instantiate the InferenceDataset and DataLoader
inference_dataset = InferenceDataset(INFERENCE_IMAGE_DIR, IMAGE_HEIGHT, IMAGE_WIDTH)
inference_dataloader = DataLoader(
    inference_dataset,
    batch_size=1, # Typically batch size of 1 for inference
    shuffle=False,
    num_workers=CPUS,
    collate_fn=collate_fn # Use collate_fn
)
################################################################################



###############################################################################
# 4. Load the Trained Model
VIS_SCORE_THRESHOLD = .5
print(f"\nLoading trained model from {MODEL_SAVE_PATH} for inference...")
loaded_model = get_ssd_model(num_classes=NUM_CLASSES, score_thresh=VIS_SCORE_THRESHOLD) 
loaded_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
loaded_model.to(device)
loaded_model.eval() # Set to evaluation mode
print("Model loaded successfully for inference.")
#############################################################################

###############################################################################
print("\n--- Example Prediction on a Validation Sample (from Training Data) ---")
#####
# Load presaved dataset
dataset = torch.load(SAVE_PATH,weights_only=False)
# --- Split Dataset into Training and Validation ---
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")

#model.eval() # Set model to evaluation mode
loaded_model.eval()#?
with torch.no_grad():
    # Get one sample from the validation dataset
    sample_image_tensor, sample_target = val_dataset[0] 
    
    # Add batch dimension and move to device
    sample_image_input = [sample_image_tensor.to(device)]
    
    # Get model prediction (list of dicts)
    #predictions = model(sample_image_input) 
    predictions = loaded_model(sample_image_input)
    
    
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

    fig, axes = plt.subplots(1, 1, figsize=(6,6), tight_layout=True)
    axes.imshow(sample_image_np,origin='lower',aspect='auto',zorder=-5)
    #axes.set_title(f'Validation Sample: Input Image with True and Predicted BBoxes (Model Threshold: {VIS_SCORE_THRESHOLD})')
    #axes.axis('off')

    # Draw all true bounding boxes
    for i, bbox in enumerate(true_boxes):
        draw_bbox(axes, bbox, color='green', label='True Feature' if i == 0 else None, width=2)
    
    # Draw all predicted bounding boxes (filtered by the model's internal threshold)
    for i, bbox in enumerate(predicted_boxes):
        score = predicted_scores[i]
        label = predicted_labels[i]
        draw_bbox(axes, bbox, color='red', label=f'Proposed Feature' if i<3 else None, score=score, width=2)

    axes.set_xlabel('Image-X');axes.set_ylabel('Image-Y')
    axes.set_rasterization_zorder(-1)
    #plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig('Training_Validation.pdf',transparent=True)






# 5. Run Inference and Visualize Predictions
print("\nRunning inference on new images...")
for i, (image_tensors, img_paths) in enumerate(inference_dataloader):
    print(f"Processing image: {os.path.basename(img_paths[0])}")
    
    image_tensors_on_device = [img.to(device) for img in image_tensors]
    
    with torch.no_grad():
        predictions = loaded_model(image_tensors_on_device) 
    
    if predictions and 'scores' in predictions[0]:
        print(f"Raw predicted scores for inference image: {predictions[0]['scores'].cpu().numpy()}")
    else:
        print("No predictions or scores found for inference image (all below internal threshold or no detections).")

    predicted_boxes = predictions[0]['boxes'].cpu().numpy() if predictions else np.array([])
    predicted_labels = predictions[0]['labels'].cpu().numpy() if predictions else np.array([])
    predicted_scores = predictions[0]['scores'].cpu().numpy() if predictions else np.array([])


    # Denormalize image for display
    image_np = image_tensors[0].cpu().permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 1, figsize=(6,6),tight_layout=True)
    #axes.imshow(image_np,extent=[.75,1.1,0,650])
    
    Z_map = np.sum(image_np,axis=2)[::-1] 
    map_x = np.linspace(.75,1.1,128)
    map_y = np.linspace(0,650,128)
    axes.contourf(map_x, map_y, Z_map,cmap='viridis',levels=100,zorder=-5)
    axes.set_xlabel('Time [s]')
    axes.set_ylabel('Freq [kHz]')
    axes.set_rasterization_zorder(-1)
    #axes.set_title(f'Inference: {os.path.basename(img_paths[0])} with Predicted BBoxes (Model Threshold: {VIS_SCORE_THRESHOLD})')
    #axes.axis('off')
    
    for k, bbox in enumerate(predicted_boxes):
        bbox[bbox>=128] = 127
        score = predicted_scores[k]
        label = predicted_labels[k]
        draw_bbox(axes, bbox, color='red', label=f'Class {label}', 
                  score=score, width=2,map_x=map_x,map_y=map_y)
            
    #plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(INFERENCE_IMAGE_DIR_SAVE+'Training_Prediction_%d.pdf'%i,transparent=True)

print("\nInference on new images complete.")
print(f"You can find the generated inference images in the '{INFERENCE_IMAGE_DIR}' directory.")
