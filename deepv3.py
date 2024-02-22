import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import os

# Define paths to your dataset
dataset_dir = 'C:\\Users\Defargobeze\Desktop\Ambo\Dataset'

# Define image and label directories
image_dir = os.path.join(dataset_dir, 'images/A')
label_dir = os.path.join(dataset_dir, 'images/A')

# Define input image size
input_shape = (224, 224, 3)

# Define number of classes
num_classes = 3  # Change this according to the number of classes in your dataset

# Load images and labels
images = []
labels = []

for filename in os.listdir(image_dir):
    img = load_img(os.path.join(image_dir, filename), target_size=input_shape[:2])
    img_array = img_to_array(img)
    images.append(img_array)
    
    label_img = load_img(os.path.join(label_dir, filename), target_size=input_shape[:2], color_mode='grayscale')
    label_array = img_to_array(label_img)
    labels.append(label_array)

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

print(images)
print(labels)
# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Preprocess input images
X_train = preprocess_input(X_train)
X_val = preprocess_input(X_val)

# Define the DeepLabV3 model
base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
base_model.trainable = False

# Add a classification head
x = layers.Conv2D(num_classes, (1, 1), activation='softmax')(base_model.output)

# Define the model
model = Model(inputs=base_model.input, outputs=x)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss=SparseCategoricalCrossentropy(),
              metrics=['accuracy', MeanIoU(num_classes=num_classes)])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model
loss, accuracy, mean_iou = model.evaluate(X_val, y_val)
print(f'Loss: {loss}, Accuracy: {accuracy}, Mean IoU: {mean_iou}')

# Save the model
model.save('deeplabv3_model.h5')
