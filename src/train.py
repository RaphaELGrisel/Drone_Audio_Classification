import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from data_processing import DataProcessing
from model import Model


class Train():
    def __init__(self,data_dir,n_epochs):
        self.data_path = data_dir
        self.n_epochs = n_epochs
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def datasets(self,train_split=0.7,test_split=0.15,val_split=0.15):

        dataset = DataProcessing(self.data_path)
        spectro_dataset = dataset.get_spectrogram_dataset()
        spectro_list = list(spectro_dataset)
        total_size = len(spectro_list)

        train_size = int(train_split * total_size)
        val_size = int(val_split * total_size)
        test_size = total_size - train_size - val_size  # Évite les erreurs d'arrondi

        # Découpage du dataset
        train_ds = tf.data.Dataset.from_tensor_slices(spectro_list[:train_size])
        val_ds = tf.data.Dataset.from_tensor_slices(spectro_list[train_size:train_size + val_size])
        test_ds = tf.data.Dataset.from_tensor_slices(spectro_list[train_size + val_size:])

        # Optimisation des datasets
        self.train_ds = train_ds.cache().shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
        self.val_ds = val_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)
        self.test_ds = test_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)

        return train_ds, val_ds, test_ds
    

    def training_loop(self):
        model1 = Model(self.train_ds)
        model = model1.CNN(2)
        model.compile(
            optimizer = tf.keras.optimizers.Adam(),
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        history = model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.n_epochs,
            callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2)
        )
        
        
        self.plot_training_results(history)
        return model
    
    def plot_training_results(self,history):

        metrics = history.history
        
        plt.figure(figsize=(16,6))
        plt.subplot(1,2,1)
        plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
        plt.legend(['loss', 'val_loss'])
        plt.ylim([0, max(plt.ylim())])
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
        model = keras.models.load_model(path)
        print(f"Modèle chargé de puis {path}")
        return model






