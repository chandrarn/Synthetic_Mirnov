import numpy as np
from PIL import Image, ImageDraw
import random
from scipy.ndimage import gaussian_filter
import math

# --- Configuration Parameters (can be adjusted) ---
IMAGE_HEIGHT = 200
IMAGE_WIDTH = 200

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
NUM_SQUIGGLES_PER_IMAGE = (0, 3) # Can have no squiggles
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

# Example usage (for testing this script independently)
if __name__ == '__main__':
    import matplotlib.pyplot as plt # Import here for independent testing
    img, bboxes = generate_image_and_individual_bboxes(IMAGE_HEIGHT, IMAGE_WIDTH)
    
    # Draw bounding boxes on a copy for visualization
    img_display = img.copy()
    draw_display = ImageDraw.Draw(img_display)
    for bbox in bboxes:
        draw_display.rectangle(bbox, outline="red", width=2)
    plt.close('Test_Image')
    plt.figure('Test_Image')
    plt.imshow(img_display)
    plt.title("Generated Image with BBoxes")
    plt.axis('off')
    plt.show()
