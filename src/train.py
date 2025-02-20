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

    def datasets(self,train_split=0.7,test_split=0.15,val_split=0.15):

        dataset = DataProcessing(self.data_path)
        spectro_dataset = dataset.get_spectrogram_dataset()
        x = spectro_dataset.take(1)
        train_size = int(train_split*len(x))
        val_size = int(val_split*len(x))
        test_size = int(test_split*len(x))

        train_ds = spectro_dataset.take(train_size)
        val_ds = spectro_dataset.take(val_size)
        test_ds = spectro_dataset.take(test_size)

        train_ds = train_ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)

        return train_ds, val_ds, test_ds
    

    def training_loop(self,train_ds,val_ds):
        model1 = Model(train_ds)
        model = model1.CNN(2)
        model.compile(
            optimizer = tf.keras.optimizes.Adam(),
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        n = self.n_epochs
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=n,
            callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2)
        )
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

        return model 
    
    @staticmethod
    def save_model(trained_model,model_name,saving_path="saved_model"):
        path = os.path.join(saving_path,model_name)
        trained_model.save(path)

    @staticmethod
    def load_model(model_name,saving_path="saved_model"):
        path = os.path.join(saved_model,model_name)
        model = keras.models.load_model(path)
        return model






