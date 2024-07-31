import numpy as np 
import cv2
import tensorflow as tf 
import matplotlib.pyplot as plt
import os
current_dir = os.getcwd()
print(current_dir)


from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam

print("Tensorflow version: ", tf.__version__)

# working with base model
base_model = MobileNet( input_shape=(224,224,3), include_top=False)

for layer in base_model.layers:
    layer.trainable = False

x = Flatten()(base_model.output)
x = Dense(units=7 , activation='softmax' )(x)

#creating model
model = Model(base_model.input, x)

model.compile(optimizer='adam', loss= categorical_crossentropy , metrics=['accuracy']  )

train_datagen = ImageDataGenerator(
    zoom_range = 0.2,
    shear_range = 0.2,
    horizontal_flip = True,
    rescale = 1./255
)

train_data = train_datagen.flow_from_directory(directory = r"C:\Git\emotion_recog\emotion_recog\emotion_recog_env\trainingdata\train",
                                               target_size = (224,224),
                                               batch_size = 32,
                                )

train_data.class_indices
