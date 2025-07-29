#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 16:35:15 2025

@author: rianc
"""


from pyTorch_SSD_lite import get_ssd_model, InferenceDataset, DataLoader,\
    CPUS, collate_fn, IMAGE_HEIGHT, IMAGE_WIDTH, MODEL_SAVE_PATH,\
    torch, NUM_CLASSES, nn, np, plt, os, draw_bbox, save_Ext


# Move model to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


INFERENCE_IMAGE_DIR = '../output_plots/training_plots/test_images/test_again/'
INFERENCE_IMAGE_DIR = 'new_inference_images'
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
VIS_SCORE_THRESHOLD = .3
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
    plt.savefig(INFERENCE_IMAGE_DIR+'/Validation/'+'Training_Prediction_%d%s.png'%(i,save_Ext))

print("\nInference on new images complete.")
print(f"You can find the generated inference images in the '{INFERENCE_IMAGE_DIR}' directory.")
