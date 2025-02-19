"""
data prerpocessing
"""
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from IPython import display
from scipy.io import wavfile

class DataProcessing():
    def __init__(self, dirpath):
        self.dataset_dir = dirpath

    
    def plot_waveform(self,n,type="yes_drone"):
        """
        Waveform plots for n audio recording
        """
        for class_name in os.listdir(self.dataset_dir):
            if class_name==type:
                class_path = os.path.join(self.dataset_dir,class_name)
                ct = 0
                plt.figure(figsize=(16,10))
                for file_name in os.listdir(class_path):
                    while ct <n:
                        audio_path = os.path.join(class_path,file_name)
                        sample_rate, audio = wavfile.read(audio_path)
                        plt.subplot(1,n,ct+1)
                        plt.plot(audio)
                        plt.title(class_name)
                        plt.grid(True)
                    ct+=1
                plt.show()


