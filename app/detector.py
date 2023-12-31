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

if video_source.isdecimal():
    video_source = int(video_source)
elif video_source.endswith('.mp4'):
    pass
else:
    print('Incorrect source')
    exit()

cap = cv2.VideoCapture(video_source)

while cap.isOpened():
    _, frame = cap.read()
    resized_frame = np.expand_dims(tf.image.resize(frame, (120, 120)) / 255.0, 0)

    y_hat = model.predict(resized_frame, verbose='0')
    dog_prob = y_hat[0]
    bbox = y_hat[1][0]

    if dog_prob > 0.5:
        cv2.rectangle(
            frame,
            tuple(np.multiply(bbox[:2], [frame.shape[1], frame.shape[0]]).astype(int)),
            tuple(np.multiply(bbox[2:], [frame.shape[1], frame.shape[0]]).astype(int)),
            (255, 0, 0), 1
        )
        
        cv2.rectangle(
            frame,
            tuple(np.add(np.multiply(bbox[:2], [frame.shape[1], frame.shape[0]]).astype(int), [0, -30])),
            tuple(np.add(np.multiply(bbox[:2], [frame.shape[1], frame.shape[0]]).astype(int), [80, 0])),
            (255, 0, 0), -1
        )

        cv2.putText(
            frame,
            'dog',
            tuple(np.add(np.multiply(bbox[:2], [frame.shape[1], frame.shape[0]]).astype(int), [0, -5])),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (255, 255, 255), 2, cv2.LINE_AA
        )

    cv2.imshow('Dog tracker', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


