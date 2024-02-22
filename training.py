from tensorflow.keras.applications import DeepLabV3
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import datasetPrep as dsp

# Create DeepLabV3 model
input_tensor = Input(shape=(dsp.image_size[0], dsp.image_size[1], 3))
model = DeepLabV3(input_tensor=input_tensor, classes=dsp.num_classes, backbone='xception', weights='pascal_voc')

# Compile the model
model.compile(optimizer=Adam(lr=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(dsp.images, dsp.masks, batch_size=8, epochs=10, validation_split=0.2)
