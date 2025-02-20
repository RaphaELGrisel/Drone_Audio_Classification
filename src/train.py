import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from data_processing import DataProcessing
from model import Model


class Train():
    def __init__(self,data_dir):
        self.data_path = data_dir

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
