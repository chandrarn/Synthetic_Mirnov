#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 18:24:50 2025

@author: rianc
"""


# Load Libraries
import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
import tensorflow_datasets as tfds
from IPython.display import clear_output
import matplotlib.pyplot as plt

import os
import glob
# Import data
dataset, info = tfds.load('oxford_iiit_pet:4.*.*', with_info=True)

# Data normalizer
def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask

# Prep training, testing datasets
@tf.function
def load_image_train(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

  if tf.random.uniform(()) > 0.5:
    input_image = tf.image.flip_left_right(input_image)
    input_mask = tf.image.flip_left_right(input_mask)

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask

def load_image_test(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))
    
    input_image, input_mask = normalize(input_image, input_mask)
    
    return input_image, input_mask


#######################################
# Split train, test
TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 64
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE


train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
test = dataset['test'].map(load_image_test)

train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset = test.batch(BATCH_SIZE)

##############################
# Show image, mask
def display(display_list,title = ['Input Image', 'True Mask', 'Predicted Mask']):
    plt.figure(figsize=(15, 15))
    
    
    
    for i in range(len(display_list)):
      plt.subplot(1, len(display_list), i+1)
      plt.title(title[i])
      plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
      plt.axis('off')
    plt.show()

for image, mask in train.take(3):
  sample_image, sample_mask = image, mask
display([sample_image, sample_mask])

#################################
# Three classification categories: background, boundary, target
OUTPUT_CHANNELS = 3

#####################
# Pre-trained encoder [?]
base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

down_stack.trainable = False


########################
# Decoder or upsampler set of blocks
up_stack = [
   pix2pix.upsample(512, 3),  # 4x4 -> 8x8
   pix2pix.upsample(256, 3),  # 8x8 -> 16x16
   pix2pix.upsample(128, 3),  # 16x16 -> 32x32
   pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]
##########################
# Define U-Net Model
def unet_model(output_channels):
    inputs = tf.keras.layers.Input(shape=[128, 128, 3])
    
    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])
    
    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
      x = up(x)
      concat = tf.keras.layers.Concatenate()
      x = concat([x, skip])
    
    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 3, strides=2,
        padding='same')  #64x64 -> 128x128
    
    x = last(x)
    
    return tf.keras.Model(inputs=inputs, outputs=x)


#############################
# Compile model
model = unet_model(OUTPUT_CHANNELS)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
# Show model structure
tf.keras.utils.plot_model(model, show_shapes=True)
#############################################
# Untrained model
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


###############################################################
# --- New function to show predictions ---
def show_predictions(dataset=None, num=1):
    """
    Shows the input image, true mask, and predicted mask for a given number of examples.

    Args:
        dataset: The TensorFlow dataset to draw examples from (e.g., test_dataset).
                 If None, uses a single example from the test_dataset iterator.
        num: The number of examples to display predictions for.
             Only applicable if 'dataset' is provided.
    """
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
    else:
        # Get a single example from the test_dataset for prediction
        for image, mask in test_dataset.take(1):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])

# Example usage:
# Show predictions for a few examples from the test set
# show_predictions(test_dataset, num=3)

# To show prediction on a single, arbitrary image from the test_dataset:
show_predictions()


###############################################################################
###############################################################################
import os
import glob

# Define target image dimensions (must match model's expected input)
IMG_HEIGHT = 128
IMG_WIDTH = 128

# --- NEW: Function to load and preprocess a single image and mask from file paths ---
def load_and_preprocess_custom_image_and_mask(image_path, mask_path=None):
    """
    Loads an image and an optional mask from file paths, resizes, and normalizes them.
    """
    # Load and preprocess image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3) # Assuming RGB images
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = tf.cast(image, tf.float32) / 255.0 # Normalize to [0,1]

    if mask_path:
        # Load and preprocess mask (if provided)
        mask = tf.io.read_file(mask_path)
        # Assuming your masks are grayscale with 1 channel and values 0, 1, 2 for classes
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.image.resize(mask, [IMG_HEIGHT, IMG_WIDTH], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # IMPORTANT: Match the normalization for your original masks
        # Original masks had values 1, 2, 3 and then `mask -= 1` made them 0, 1, 2.
        # So, if your new masks are 0, 1, 2, you might not need `mask -= 1`.
        # If your new masks are 1, 2, 3, then you do need `mask -= 1`.
        # For simplicity, let's assume your new masks (if present) are also 1,2,3 for class IDs and need adjustment.
        mask = tf.cast(mask, tf.uint8) # Or tf.int32 depending on your labels
        mask -= 1 # Apply the same normalization as the original dataset

        return image, mask
    else:
        return image

# --- NEW: Load your custom test images and (optionally) masks ---
def create_custom_test_dataset(image_folder, mask_folder=None, batch_size=BATCH_SIZE):
    image_paths = sorted(glob.glob(os.path.join(image_folder, '*.png')))
    
    if mask_folder:
        mask_paths = sorted(glob.glob(os.path.join(mask_folder, '*.png')))
        if len(image_paths) != len(mask_paths):
            raise ValueError("Number of images and masks do not match!")
        
        # Create dataset from image and mask paths
        custom_dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
        custom_dataset = custom_dataset.map(load_and_preprocess_custom_image_and_mask, 
                                            num_parallel_calls=tf.data.AUTOTUNE)
    else:
        # Create dataset from image paths only (for prediction without ground truth)
        custom_dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        custom_dataset = custom_dataset.map(lambda img_p: load_and_preprocess_custom_image_and_mask(img_p),
                                            num_parallel_calls=tf.data.AUTOTUNE)
        
    custom_dataset = custom_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return custom_dataset

# Example Usage:
# IMPORTANT: Replace 'path/to/your/new_test_data/images' and 'path/to/your/new_test_data/masks'
# with the actual paths to your new test image and mask folders.
# If you don't have masks, omit the mask_folder argument.

# Scenario 1: You have both new images and their corresponding masks
new_image_folder = '../output_plots/training_plots/test_images/'#'path/to/your/new_test_data/images'
new_mask_folder = None#'path/to/your/new_test_data/masks'
new_test_dataset = create_custom_test_dataset(new_image_folder, new_mask_folder)

# Scenario 2: You only have new images (no ground truth masks for evaluation)
# new_image_folder_only = 'path/to/your/new_test_data/images_only'
# new_test_dataset_no_masks = create_custom_test_dataset(new_image_folder_only)


# ... (your existing model definition and compilation) ...

###############################################################
# --- New function to show predictions ---
# (This function is already good, we just need to pass the new dataset to it)
def show_predictions(dataset=None, num=1,title=['Input Image','Zero-Training Prediction']):
    """
    Shows the input image, true mask, and predicted mask for a given number of examples.

    Args:
        dataset: The TensorFlow dataset to draw examples from (e.g., test_dataset or new_test_dataset).
                 If None, uses a single example from the test_dataset iterator.
        num: The number of examples to display predictions for.
             Only applicable if 'dataset' is provided.
    """
    if dataset:
        for image in dataset.take(num):
            mask = None
            # If mask is None (when loading only images), handle it for display
            if mask is None: # This check is crucial if you load images without masks
                 pred_mask = model.predict(image)
                 # print(create_mask(pred_mask))
                 #pred_mask[pred_mask<2] = 0
                 # Only display input image and predicted mask
                 display([image[0], create_mask(pred_mask)],title)
            else:
                 pred_mask = model.predict(image)
                 display([image[0], mask[0], create_mask(pred_mask)])
    else:
        # Get a single example from the test_dataset for prediction
        for image, mask in test_dataset.take(1):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])

# Example usage with your new test set:

# If you loaded images with masks:
print("\n--- Showing predictions on new custom test dataset (with masks) ---")
show_predictions(new_test_dataset, num=3) # Show 3 examples from your new dataset

# If you loaded images WITHOUT masks (and want to see predictions):
# print("\n--- Showing predictions on new custom test dataset (no masks) ---")
# show_predictions(new_test_dataset_no_masks, num=3) # Display input and predicted only

# You can also evaluate your model on this new dataset if it has masks:
# If you have masks in your new_test_dataset:
# loss, accuracy = model.evaluate(new_test_dataset)
# print(f"Loss on new test data: {loss}, Accuracy on new test data: {accuracy}")


# Define callback function to output model predictions
class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    # show_predictions()
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))
EPOCHS = 10
VAL_SUBSPLITS = 5
VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS

# dataset_special = gen_special_test_set()

# show_predictions(dataset_special, 2)
# Run Model
model_history = model.fit(train_dataset, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_dataset,
                          callbacks=[DisplayCallback()])

show_predictions(new_test_dataset, num=3,title=['Input Image','Trained Prediction'])