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

# IMPORTS for Distributed Training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# Import SSDLite320_MobileNet_V3_Large model
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
# NEW: Import specific SSD head components for replacement
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead, SSDLiteRegressionHead
from torchvision.models._utils import _ovewrite_named_param
INFERENCE_IMAGE_DIR = 'new_inference_images'

###############################
from dataset_builder_SSD import generate_image_and_individual_bboxes

# --- Configuration Parameters ---
IMAGE_HEIGHT = 250
IMAGE_WIDTH = 250
NUM_SAMPLES = 1000 # Increased samples for better training
SAVE_PATH = 'synthetic_multi_bbox_dataset_ssd.pth' # New save path for dataset
MODEL_SAVE_PATH = 'ssd_mobilenet_v3_model.pth' # Path to save the trained SSD model
NUM_EPOCHS = 15
NUM_CLASSES = 2

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
    # 1. Load the pre-trained SSDLite model with its DEFAULT configuration (num_classes=91 for COCO)
    #    This ensures the entire backbone and original heads are correctly instantiated with matching weights,
    #    and specifically that the `backbone.extra` layers are untouched.
    model = ssdlite320_mobilenet_v3_large(weights="DEFAULT")

    # 2. Get the necessary parameters to create new classification and regression heads.
    #    These parameters are from the *original* head, which is correctly configured.
    #    Access in_channels from the original classification head directly.
    in_channels_list = model.head.classification_head.in_channels
    num_anchors_per_location = model.anchor_generator.num_anchors_per_location()

    # 3. Create new classification and regression heads with our custom `num_classes`.
    new_classification_head = SSDLiteClassificationHead(
        in_channels_list=in_channels_list,
        num_classes=num_classes, # Our custom NUM_CLASSES (e.g., 2)
        num_anchors_per_location=num_anchors_per_location
    )
    new_regression_head = SSDLiteRegressionHead(
        in_channels_list=in_channels_list,
        num_anchors_per_location=num_anchors_per_location
    )

    # 4. Replace the original heads in the pre-trained model with our new heads.
    model.head.classification_head = new_classification_head
    model.head.regression_head = new_regression_head

    # 5. Set the inference score threshold for the model.
    _ovewrite_named_param(model, "score_thresh", score_thresh)

    return model

# --- Main Training and Inference Logic (Wrapped for DDP) ---

# All the existing training, saving, loading, and inference logic
# will be encapsulated in a main function that gets called by each DDP process.
def main_worker(rank, world_size, dataset, val_dataset, VIS_SCORE_THRESHOLD):
    # 1. Initialize the distributed environment
    # Use 'gloo' backend for CPU distributed training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355' # Choose an unused port
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
    
    # 2. Set device for this process to CPU
    device = torch.device('cpu')
    print(f"Process {rank}/{world_size} initialized on device: {device}")

    # 3. Create Distributed Samplers for DataLoader
    # This ensures each process gets a unique part of the data.
    train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    # 4. Create DataLoaders (adjust batch size for single CPU core in mind)
    BATCH_SIZE_PER_CORE = 2 # Adjust this. Total batch size will be BATCH_SIZE_PER_CORE * world_size
    train_dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE_PER_CORE,
        shuffle=False, # Shuffle handled by sampler
        num_workers=2, 
        collate_fn=collate_fn,
        sampler=train_sampler
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE_PER_CORE,
        shuffle=False, # Shuffle handled by sampler
        num_workers=2, 
        collate_fn=collate_fn,
        sampler=val_sampler
    )

    # 5. Instantiate the model and move to device
    model = get_ssd_model(num_classes=NUM_CLASSES, score_thresh=VIS_SCORE_THRESHOLD)
    model.to(device)

    # 6. Wrap the model with DistributedDataParallel
    model = DDP(model, device_ids=None) # device_ids=None for CPU DDP

    # Optimizer (keep as is)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    # --- Training Loop ---
    
    print(f"\n--- Process {rank}: Starting Training for {NUM_EPOCHS} Epochs ---")

    train_losses = []
    val_losses = []

    for epoch in range(NUM_EPOCHS):
        # Set epoch for DistributedSampler to ensure proper shuffling each epoch
        train_sampler.set_epoch(epoch)

        model.train() # Set model to training mode
        running_loss = 0.0
        for batch_idx, (images, targets) in enumerate(train_dataloader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            if not torch.isfinite(losses):
                print(f"Process {rank} Warning: Non-finite loss in epoch {epoch+1}, batch {batch_idx}. Skipping backward pass.")
                continue

            losses.backward()
            optimizer.step()

            # For accurate total loss across all CPUs, you would typically reduce the loss
            # across processes using dist.all_reduce(). For simplicity here, we accumulate local loss.
            running_loss += losses.item() * len(images)

        epoch_train_loss = running_loss / len(train_dataloader.dataset)
        train_losses.append(epoch_train_loss)
        # Only print from rank 0 to avoid cluttered output
        if rank == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {epoch_train_loss:.6f}")

        # --- Validation Loop ---
        # Temporarily set to train mode to get loss dict output, then switch back to eval mode.
        model.train() 
        val_running_loss = 0.0
        with torch.no_grad():
            for images, targets in val_dataloader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                if not torch.isfinite(losses):
                    print(f"Process {rank} Warning: Non-finite validation loss in epoch {epoch+1}. Skipping batch.")
                    continue

                val_running_loss += losses.item() * len(images)
                
        model.eval() # Switch back to eval mode after loss calculation
        
        epoch_val_loss = val_running_loss / len(val_dataloader.dataset)
        val_losses.append(epoch_val_loss)
        if rank == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Validation Loss: {epoch_val_loss:.6f}")

    if rank == 0: # Only save from rank 0 to avoid multiple saves
        print("\n--- Training Complete ---")
        print(f"\nSaving trained model to {MODEL_SAVE_PATH}...")
        torch.save(model.module.state_dict(), MODEL_SAVE_PATH) # Access underlying model with .module
        print("Model state dictionary saved successfully.")

        # --- Visualize Training and Validation Loss (only from rank 0) ---
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label='Training Loss')
        plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs (SSDLite MobileNetV3)')
        plt.legend()
        plt.grid(True)
        plt.show()

        # --- Example Prediction on a Validation Sample (from Training Data) ---
        print("\n--- Example Prediction on a Validation Sample (from Training Data) ---")
        model.eval() 
        with torch.no_grad():
            sample_image_tensor, sample_target = val_dataset[0] 
            sample_image_input = [sample_image_tensor.to(device)]
            predictions = model(sample_image_input) 
            
            if predictions and 'scores' in predictions[0]:
                print(f"\nRaw predicted scores for the validation sample: {predictions[0]['scores'].cpu().numpy()}")
            else:
                print("No predictions or scores found for the validation sample (all below internal threshold or no detections).")

            predicted_boxes = predictions[0]['boxes'].cpu().numpy() if predictions else np.array([])
            predicted_labels = predictions[0]['labels'].cpu().numpy() if predictions else np.array([])
            predicted_scores = predictions[0]['scores'].cpu().numpy() if predictions else np.array([])

            true_boxes = sample_target['boxes'].cpu().numpy()
            true_labels = sample_target['labels'].cpu().numpy()

            sample_image_np = sample_image_tensor.cpu().permute(1, 2, 0).numpy()

            fig, axes = plt.subplots(1, 1, figsize=(8, 8))
            axes.imshow(sample_image_np)
            axes.set_title(f'Validation Sample: Input Image with True and Predicted BBoxes (Model Threshold: {VIS_SCORE_THRESHOLD})')
            axes.axis('off')

            for i, bbox in enumerate(true_boxes):
                draw_bbox(axes, bbox, color='green', label=f'True' if i == 0 else None, width=2)
            
            for i, bbox in enumerate(predicted_boxes):
                score = predicted_scores[i]
                label = predicted_labels[i]
                draw_bbox(axes, bbox, color='red', label=f'Pred', score=score, width=2)

            plt.legend()
            plt.tight_layout()
            plt.show()

    # Clean up distributed environment
    dist.destroy_process_group()


# --- Helper for drawing bounding boxes ---
def draw_bbox(ax, bbox, color='red', label=None, score=None, width=2, image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT):
    x_min, y_min, x_max, y_max = bbox
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


# --- Entry Point for Multi-CPU Training ---
if __name__ == '__main__':
    # Initial dataset loading (can be done once by a single process)
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

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    VIS_SCORE_THRESHOLD = 0.01

    # Get number of available CPU cores
    # It's recommended to use a subset of cores in real-world scenarios to avoid resource contention
    # For demonstration, we'll use all detected CPU cores.
    world_size = os.cpu_count()
    if world_size is None or world_size < 1:
        print("Could not detect CPU cores or found 0. Defaulting to 1 process.")
        world_size = 1
    
    print(f"Found {world_size} CPU cores. Launching distributed training.")
    # Launch main_worker function as multiple processes, one for each CPU core
    mp.spawn(main_worker,
             args=(world_size, dataset, val_dataset, VIS_SCORE_THRESHOLD),
             nprocs=world_size,
             join=True)

    # --- Inference Section (runs after distributed training completes on rank 0) ---
    # This section remains outside the main_worker because it should only run once
    # after the model is trained and saved.

    print("\n--- Testing with New, Unseen Images ---")

    
    os.makedirs(INFERENCE_IMAGE_DIR, exist_ok=True)

    num_inference_images_to_generate = 5
    print(f"Generating {num_inference_images_to_generate} dummy images for inference in '{INFERENCE_IMAGE_DIR}'...")
    for i in range(num_inference_images_to_generate):
        img_pil, _ = generate_image_and_individual_bboxes(IMAGE_HEIGHT, IMAGE_WIDTH) 
        img_pil.save(os.path.join(INFERENCE_IMAGE_DIR, f'inference_img_{i+1}.png'))
    print("Dummy inference images generated.")

    inference_dataset = InferenceDataset(INFERENCE_IMAGE_DIR, IMAGE_HEIGHT, IMAGE_WIDTH)
    inference_dataloader = DataLoader(
        inference_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    print(f"\nLoading trained model from {MODEL_SAVE_PATH} for inference...")
    loaded_model = get_ssd_model(num_classes=NUM_CLASSES, score_thresh=VIS_SCORE_THRESHOLD) 
    # map_location='cpu' is explicitly set here as the inference will run on CPU
    loaded_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location='cpu'))
    loaded_model.to('cpu') # Move model to CPU
    loaded_model.eval()
    print("Model loaded successfully for inference.")

    print("\nRunning inference on new images...")
    for i, (image_tensors, img_paths) in enumerate(inference_dataloader):
        print(f"Processing image: {os.path.basename(img_paths[0])}")
        
        image_tensors_on_device = [img.to('cpu') for img in image_tensors] # Move to CPU
        
        with torch.no_grad():
            predictions = loaded_model(image_tensors_on_device) 
        
        if predictions and 'scores' in predictions[0]:
            print(f"Raw predicted scores for inference image: {predictions[0]['scores'].cpu().numpy()}")
        else:
            print("No predictions or scores found for inference image (all below internal threshold or no detections).")

        predicted_boxes = predictions[0]['boxes'].cpu().numpy() if predictions else np.array([])
        predicted_labels = predictions[0]['labels'].cpu().numpy() if predictions else np.array([])
        predicted_scores = predictions[0]['scores'].cpu().numpy() if predictions else np.array([])

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

print("\nInference on new images complete.")
print(f"You can find the generated inference images in the '{INFERENCE_IMAGE_DIR}' directory.")
