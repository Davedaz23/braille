import os
import numpy as np
import PIL.Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Set path to your dataset directory
dataset_dir = 'C:\\Users\Defargobeze\Desktop\Ambo\Dataset'

# Define image dimensions
image_height, image_width = 128, 128

# Initialize lists to store images and labels
X_data, y_data = [], []

# Iterate through each class directory
for class_name in os.listdir(dataset_dir):
    class_dir = os.path.join(dataset_dir, class_name)
    if os.path.isdir(class_dir):
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            if image_path.endswith('.jpg') or image_path.endswith('.png'):
                # Load and preprocess image
                image = load_img(image_path, target_size=(image_height, image_width))
                image = img_to_array(image) / 255.0  # Normalize pixel values
                X_data.append(image)
                y_data.append(class_name)

# Convert lists to numpy arrays
X_data = np.array(X_data)
y_data = np.array(y_data)

# Split dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# Print shapes of training and validation sets
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)
