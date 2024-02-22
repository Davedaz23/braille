#!pip install tensorflow

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import MeanIoU
import os
from sklearn.model_selection import train_test_split
from glob import glob
# Check if GPU is available
if tf.test.gpu_device_name():
    print('GPU is available')
else:
    print('GPU is not available')

# Define paths to your dataset
train_images_path = 'C:\Users\Defargobeze\Desktop\Ambo\Dataset\images\A'
train_masks_path = 'C:\Users\Defargobeze\Desktop\Ambo\Dataset\masks\A'

train_image_files = glob(os.path.join(train_images_path, '*.jpg'))
train_mask_files = glob(os.path.join(train_masks_path, '*.jpg'))


train_images, val_images, train_masks, val_masks = train_test_split(train_image_files, train_mask_files, test_size=0.2, random_state=42)

# Define image dimensions and batch size
IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 8

# Function to load and preprocess images and masks
def parse_image(img_path, mask_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, [IMG_HEIGHT, IMG_WIDTH])
    mask = tf.cast(mask, tf.int32)

    return img, mask

# Create dataset from images and masks
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_masks))
train_dataset = train_dataset.map(parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_masks))
val_dataset = val_dataset.map(parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# Define the DeepLabV3 model
base_model = MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False, weights='imagenet')
x = base_model.output
x = Conv2D(256, 3, activation='relu', padding='same')(x)
x = Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(x)
x = concatenate([base_model.get_layer('block_6_expand_relu').output, x])
x = Conv2D(64, 3, activation='relu', padding='same')(x)
x = Conv2DTranspose(32, 2, strides=(2, 2), padding='same')(x)
x = concatenate([base_model.get_layer('block_3_expand_relu').output, x])
x = Conv2D(32, 3, activation='relu', padding='same')(x)
x = Conv2DTranspose(16, 2, strides=(2, 2), padding='same')(x)
x = Conv2D(1, 3, activation='softmax', padding='same')(x)

model = Model(inputs=base_model.input, outputs=x)

# Compile the model
model.compile(optimizer=Adam(lr=1e-4), loss=SparseCategoricalCrossentropy(), metrics=[MeanIoU(num_classes=2)])

# Define callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint('deeplabv3_model.h5', save_best_only=True, monitor='val_loss', mode='min')
]

# Train the model
model.fit(train_dataset, epochs=20, validation_data=val_dataset, callbacks=callbacks)

