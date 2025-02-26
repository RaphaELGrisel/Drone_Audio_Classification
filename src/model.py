import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models


class Model():
    def __init__(self,training_dataset):
        self.train_spectrogram = training_dataset

    def CNN(self,n_labels = 2 ,input_dim=(124,129,1)):

        """
        When using Mel spectrogram, comment normalisation layer
        
        """
        input_shape = input_dim
        norm_layer = layers.Normalization()
        spectrogram_ds = self.train_spectrogram.map(lambda spec, label: spec)
        norm_layer.adapt(spectrogram_ds)
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Resizing(32,32),
            norm_layer,
            layers.Conv2D(32,3,activation="relu"),
            layers.Conv2D(64,3,activation="relu"),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(128,activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(n_labels)
        ])

        model.summary()
        return model
    
    
