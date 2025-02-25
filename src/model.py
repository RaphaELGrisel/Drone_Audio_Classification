import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models


class Model():
    def __init__(self,training_dataset):
        self.train_spectrogram = training_dataset

    def CNN(self,n_labels = 2 ,input_dim=(124,129,1)):
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
            #layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(128,activation="relu"),
            #layers.Dropout(0.5),
            layers.Dense(n_labels)
        ])

        model.summary()
        return model
    
    def CNN_complexity_1(self, n_labels=2, input_dim=(124, 129, 1)):
        input_shape = input_dim
        norm_layer = layers.Normalization()

        # Adapter la normalisation sur les données
        spectrogram_ds = self.train_spectrogram.map(lambda spec, label: spec)
        norm_layer.adapt(spectrogram_ds)

        model = models.Sequential([
            # Entrée et normalisation
            layers.Input(shape=input_shape),
            layers.Resizing(32, 32),  # Redimensionnement des spectrogrammes
            norm_layer,

            # 1er bloc convolutionnel avec BatchNorm et Dropout
            layers.Conv2D(64, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.3),

            # 2ème bloc convolutionnel avec BatchNorm et Dropout
            layers.Conv2D(128, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.3),

            # 3ème bloc convolutionnel avec BatchNorm
            layers.Conv2D(256, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),

            # Bloc résiduel pour améliorer l'apprentissage
            layers.Conv2D(256, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Add(),  # Ajout des connexions résiduelles
            layers.MaxPooling2D(pool_size=(2, 2)),

            # Couche Flatten pour connecter aux couches Fully Connected
            layers.Flatten(),

            # Couches Fully Connected (Dense)
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),

            # Sortie avec le nombre de classes
            layers.Dense(n_labels)
        ])

        model.summary()
        return model
