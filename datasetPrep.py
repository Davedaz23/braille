import os
import numpy as np
from PIL import Image

# Define paths to your dataset
dataset_dir = r'C:\Users\Defargobeze\Desktop\Ambo\Dataset'
images_dir = os.path.join(dataset_dir, 'images')  # Assume 'images' directory contains alphabet directories
masks_dir = os.path.join(dataset_dir, 'masks')

# Define image size
image_size = (256, 256)

# Initialize empty lists to store images and masks
images = []
masks = []

# Loop through alphabet directories
for alphabet_dir in os.listdir(images_dir):
    alphabet_images_dir = os.path.join(images_dir, alphabet_dir)
    alphabet_masks_dir = os.path.join(masks_dir, alphabet_dir)

    # Loop through images in the alphabet directory
    for img_file in os.listdir(alphabet_images_dir):
        img_path = os.path.join(alphabet_images_dir, img_file)
        mask_path = os.path.join(alphabet_masks_dir, img_file)  # Assuming mask filenames match image filenames
        
        # Open and resize image
        img = Image.open(img_path).resize(image_size)
        img_array = np.array(img)
        images.append(img_array)
        
        # Open and resize mask
        mask = Image.open(mask_path).resize(image_size).convert('L')
        mask_array = np.array(mask)
        masks.append(mask_array)

# Convert lists to numpy arrays
images = np.array(images)
masks = np.array(masks)
