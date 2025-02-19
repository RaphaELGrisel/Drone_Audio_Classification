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
                        plt.subplot(n,1,ct+1)
                        plt.plot(audio)
                        plt.title(class_name)
                        plt.grid(True)
                        ct+=1
                plt.show()

    
    def plot_spectogram(self,n,type="yes_drone"):
        """
        spectogram plots for n audio recording
        """
        
        fig, axes = plt.subplots(n, 2, figsize=(12, 6 * n))  # Deux colonnes : waveform + spectrogram

        for class_name in os.listdir(self.dataset_dir):
            if class_name==type:
                class_path = os.path.join(self.dataset_dir,class_name)
                ct = 0
                for file_name in os.listdir(class_path):
                    while ct <n:
                        audio_path = os.path.join(class_path, file_name)
                        sample_rate, audio = wavfile.read(audio_path)

                        # Si audio est stéréo, on garde un seul canal
                        # mono_channel doc non

                        # Calcul du spectrogramme avec STFT
                        spectrogram = tf.signal.stft(audio, frame_length=255, frame_step=128)
                        spectrogram = np.abs(spectrogram)
                        log_spec = np.log(spectrogram.T + np.finfo(float).eps)

                        # Affichage de la forme d'onde
                        timescale = np.arange(audio.shape[0]) / sample_rate
                        axes[i, 0].plot(timescale, audio)
                        axes[i, 0].set_title(f"Waveform - {file_name}")
                        axes[i, 0].set_xlabel("Temps (s)")
                        axes[i, 0].set_ylabel("Amplitude")
                        axes[i, 0].grid(True)

                        # Affichage du spectrogramme
                        X = np.linspace(0, np.size(spectrogram), num=log_spec.shape[1])
                        Y = range(log_spec.shape[0])
                        axes[i, 1].pcolormesh(X, Y, log_spec, shading="auto", cmap="viridis")
                        axes[i, 1].set_title(f"Spectrogram - {file_name}")
                        axes[i, 1].set_xlabel("Frames")
                        axes[i, 1].set_ylabel("Fréquence")
                        ct+=1

                plt.tight_layout()
                plt.show()



