import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from model import Model


class Train():
    def __init__(self,n_epochs, train, val, model):
        self.n_epochs = n_epochs
        self.train_ds = train
        self.val_ds = val
        self.model = model
    

    def training_loop(self):

        train_ds = self.train_ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
        val_ds = self.val_ds.cache().prefetch(tf.data.AUTOTUNE)

        lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps= self.n_epochs,
            decay_rate=0.96,
            staircase=False
        )
        self.model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler),
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.n_epochs,
            #callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=5)
        )
        
        
        self.plot_training_results(history)
        return self.model
    
    def plot_training_results(self,history):

        metrics = history.history
        
        plt.figure(figsize=(16,6))
        plt.subplot(1,2,1)
        plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
        plt.legend(['loss', 'val_loss'])
        plt.ylim([0, 5])
        plt.xlabel('Epoch')
        plt.ylabel('Loss [CrossEntropy]')

        plt.subplot(1,2,2)
        plt.plot(history.epoch, 100*np.array(metrics['accuracy']), 100*np.array(metrics['val_accuracy']))
        plt.legend(['accuracy', 'val_accuracy'])
        plt.ylim([0, 100])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy [%]')      
    
    @staticmethod
    def save_model(trained_model,model_name,saving_path="saved_model"):
        path = os.path.join(saving_path,model_name)
        trained_model.save(path)
        print(f"Modèle sauvegardé à : {path}")

    @staticmethod
    def load_model(model_name,saving_path="saved_model"):
        path = os.path.join(saving_path,model_name)
        model = tf.keras.models.load_model(path)
        print(f"Modèle chargé de puis {path}")
        return model






