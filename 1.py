import os
import numpy as np
import keras
import tensorflow as tf
import time
from random import sample

from tensorflow.keras import layers

def conv_block(x, filters, size = 1):
    x = layers.Conv1D(filters, kernel_size = size, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)

def deconv_block(x, filters, size = 1):
    x = layers.Conv1DTranspose(filters, kernel_size = size, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)


def mlp_block(x, filters, name) -> tf.Tensor:
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)

def get_shape_segmentation_model(num_points: int, num_classes: int) -> keras.Model:
    input_points = keras.Input(shape=(1024, 3))

    features_128_1 = conv_block(input_points, filters=128)
    features_512 = conv_block(features_128_1, filters=512)
    features_1024 = conv_block(features_512, filters=1024)
    features_1024_1 = conv_block(features_1024, filters=1024)
    features_1024_2 = deconv_block(features_1024_1, filters=1024)
    features_512_1 = deconv_block(features_1024_2, filters=512)
    features_128_4 = deconv_block(features_512_1, filters=128)
    global_features = layers.MaxPool1D(pool_size=num_points)(features_128_4)
    global_features = tf.tile(global_features, [1, num_points, 1])
    segmentation_input = layers.Concatenate()(
        [
            features_128_1,
            features_512,
            global_features,
        ]
    )
    segmentation_features = conv_block(segmentation_input, filters=128)

    outputs = layers.Conv1D(num_classes, kernel_size=1, activation="softmax")(segmentation_features)
    return keras.Model(input_points, outputs)

segmentation_model = get_shape_segmentation_model(1024, 16)
segmentation_model.summary()

segmentation_model.save('model')