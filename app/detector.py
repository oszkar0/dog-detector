import cv2
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalMaxPooling2D


def build_model():
    input = Input(shape=(120, 120, 3))
    vgg16 = VGG16(include_top=False, weights=None)(input)
    
    # classification model
    cl1 = GlobalMaxPooling2D()(vgg16)
    cl2 = Dense(2048, activation='relu')(cl1)
    cl3 = Dense(1, activation='sigmoid')(cl2)

    # bbox model
    bbox1 = GlobalMaxPooling2D()(vgg16)
    bbox2 = Dense(2048, activation='relu')(bbox1)
    bbox3 = Dense(4, activation='sigmoid')(bbox2)
    
    return Model(inputs=input, outputs=[cl3, bbox3])
    
 
arguments = sys.argv
weights_file_name = arguments[1]
video_source = arguments[2]

model = build_model()
model.load_weights(weights_file_name)