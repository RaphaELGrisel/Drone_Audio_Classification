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



    def plot_spectrogram(self, n, audio_class="yes_drone"):
        """
        Affiche les spectrogrammes et les formes d'onde pour `n` fichiers audio d'une classe donnée.

        :param n: Nombre d'échantillons à afficher.
        :param audio_class: Nom du dossier contenant les fichiers audio.
        """
        class_path = os.path.join(self.dataset_dir, audio_class)

        if not os.path.exists(class_path):
            print(f"Le dossier {audio_class} n'existe pas dans {self.dataset_dir}")
            return

        audio_files = [f for f in os.listdir(class_path) if f.endswith(".wav")]
        
        if len(audio_files) == 0:
            print(f"Aucun fichier audio trouvé dans {class_path}")
            return

        # Limiter `n` au nombre de fichiers disponibles
        n = min(n, len(audio_files))

        # Si n == 1, il faut que axes soit une dimension 1D
        fig, axes = plt.subplots(n, 2, figsize=(12, 6 * n)) if n > 1 else plt.subplots(1, 2, figsize=(12, 6))

        for i, file_name in enumerate(audio_files[:n]):
            audio_path = os.path.join(class_path, file_name)
            sample_rate, audio = wavfile.read(audio_path)

            # Si audio est stéréo, on garde un seul canal
            if len(audio.shape) > 1:
                audio = audio[:, 0]

            # Convertir l'audio en float32 et normaliser
            audio = audio.astype(np.float32)  # Conversion en float32
            audio /= np.max(np.abs(audio))  # Normalisation pour être entre -1 et 1

            # Calcul du spectrogramme avec STFT
            spectrogram = tf.signal.stft(audio, frame_length=255, frame_step=128)
            spectrogram = np.abs(spectrogram)
            log_spec = np.log(spectrogram.T + np.finfo(float).eps)

            # Affichage de la forme d'onde
            timescale = np.arange(audio.shape[0]) / sample_rate
            if n > 1:
                axes[i, 0].plot(timescale, audio)
                axes[i, 0].set_title(f"Waveform - {file_name}")
                axes[i, 0].set_xlabel("Temps (s)")
                axes[i, 0].set_ylabel("Amplitude")
                axes[i, 0].grid(True)
            else:
                axes[0].plot(timescale, audio)
                axes[0].set_title(f"Waveform - {file_name}")
                axes[0].set_xlabel("Temps (s)")
                axes[0].set_ylabel("Amplitude")
                axes[0].grid(True)

            # Affichage du spectrogramme
            X = np.linspace(0, np.size(spectrogram), num=log_spec.shape[1])
            Y = range(log_spec.shape[0])
            if n > 1:
                cax = axes[i, 1].pcolormesh(X, Y, log_spec, shading="auto", cmap="viridis")
                axes[i, 1].set_title(f"Spectrogram - {file_name}")
                axes[i, 1].set_xlabel("Frames")
                axes[i, 1].set_ylabel("Fréquence")
                fig.colorbar(cax, ax=axes[i, 1])  # Ajouter la colorbar pour chaque spectrogramme
            else:
                cax = axes[1].pcolormesh(X, Y, log_spec, shading="auto", cmap="viridis")
                axes[1].set_title(f"Spectrogram - {file_name}")
                axes[1].set_xlabel("Frames")
                axes[1].set_ylabel("Fréquence")
                fig.colorbar(cax, ax=axes[1])  # Ajouter la colorbar pour le spectrogramme

        plt.tight_layout()
        plt.show()

    @staticmethod
    def squeeze(audio, labels):
        shape = tf.shape(audio)
        if shape[-1] == 1:  # If last dimension is 1, remove it
            audio = tf.squeeze(audio, axis=-1)
        return audio, labels
    
    @staticmethod
    def get_spectrogram(waveform):
        spectrogram = tf.signal.stft(
            waveform, frame_length=255, frame_step=128
        )
        spectrogram = tf.abs(spectrogram)
        spectrogram = spectrogram[..., tf.newaxis]
        return spectrogram

    def get_spectrogram_dataset(self):
        #val_split at 0 
        audio_dataset = tf.keras.utils.audio_dataset_from_directory(
            directory = self.dataset_dir,
            batch_size=64,
            validation_split=0.0,
            seed=0,
            output_sequence_length=16000
        )
        label_names = np.array(audio_dataset.class_names)
        print("label names:",label_names)

        audio_dataset = audio_dataset.map(self.squeeze, tf.data.AUTOTUNE)
        spectro_dataset = audio_dataset.map(
            map_func=lambda audio, label: (self.get_spectrogram(audio),label),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        return spectro_dataset , label_names



