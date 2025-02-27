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
from scipy import signal , linalg
import random

import librosa
import librosa.display

import tensorflow_io as tfio

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
                        print("SHAPE:",audio.shape)
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
        audio = tf.squeeze(audio, axis=-1)
        return audio, labels
    
    @staticmethod
    def get_spectrogram(waveform):
        spectrogram = tf.signal.stft(
            waveform, frame_length=256, frame_step=128
        )
        spectrogram = tf.abs(spectrogram)
        spectrogram = spectrogram[..., tf.newaxis]
        return spectrogram
    

    """
     Using tensorflow_io

    @staticmethod
    def get_mel_spectrogram(waveform):
        spectrogram = tf.signal.stft(
            waveform, frame_length=256, frame_step=128
        )
        spectrogram = tf.abs(spectrogram)
        mel_spectro = tfio.audio.melscale(spectrogram, rate=16000, mels=129,fmin=0, fmax=8000)
        mel_spectro = tf.math.log(mel_spectro+1e-6)
        #mel_spectro = (mel_spectro - tf.reduce_min(mel_spectro)) / (tf.reduce_max(mel_spectro) - tf.reduce_min(mel_spectro))
        mel_spectro = mel_spectro[...,tf.newaxis]
        return mel_spectro
    
    """

    @staticmethod
    def get_mel_spectrogram(waveform):
        def _compute_mel_spectrogram(waveform_np):
            """
            Fonction interne exécutée en mode eager pour convertir en spectrogram
            """
            mel_spec = librosa.feature.melspectrogram(y=waveform_np, sr=16000,n_mels=129,hop_length = tf.shape(waveform)[0] // 124 + 1,n_fft =1024)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            mel_spec -= mel_spec.min()
            mel_spec /= mel_spec.max()
            return mel_spec_db.astype(np.float32)
        
        mel_spectrogram=tf.py_function(func=_compute_mel_spectrogram,inp=[waveform], Tout=tf.float32)
        mel_spectrogram.set_shape((129,124))
        return mel_spectrogram[...,tf.newaxis]
    

    @staticmethod
    def get_mel_spectrogram_tensorflow(waveform):
        spectrogram = tf.signal.stft(waveform, frame_length=1024, frame_step=256)
        magnitude_spectrogram = tf.abs(spectrogram)
        mel_filterbank = tf.signal.linear_to_mel_weight_matrix(num_mel_bins=129, num_spectrogram_bins=513, sample_rate=16000)
        mel_spectrogram = tf.tensordot(magnitude_spectrogram, mel_filterbank, axes=1)
        mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)  # Pour éviter log(0)
        return mel_spectrogram[...,tf.newaxis]


    def plot_mel_spectrogram(self,n,type="yes_drone"):
        """
        mel_spec plots for n audio recording
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
                        audio = audio.astype(np.float32)
                        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate,n_mels=129,hop_length=int((len(audio)/124)+1),n_fft=1024)
                        mel_spec -= mel_spec.min()
                        mel_spec /= mel_spec.max()
                        print(np.shape(mel_spec))
                        #print(type(mel_spec))
                        #print(type(mel_spec_db))
                        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                        librosa.display.specshow(mel_spec_db ,sr=sample_rate,x_axis="time",y_axis='mel',cmap='viridis')
                        plt.colorbar(label="dB")
                        plt.title("Mel-Spectrogramme Log")
                        plt.xlabel("Temps")
                        plt.ylabel("Echelle MEL")
                        plt.show()
                        ct+=1
                plt.show()
    
    def plot_mel_spectrogram_tensorflow(self,n,type="yes_drone"):
        for class_name in os.listdir(self.dataset_dir):
            if class_name==type:
                class_path = os.path.join(self.dataset_dir,class_name)
                ct = 0
                plt.figure(figsize=(16,10))
                for file_name in os.listdir(class_path):
                    while ct <n:
                        audio_path = os.path.join(class_path,file_name)
                        sample_rate, audio = wavfile.read(audio_path)
                        audio = audio.astype(np.float32)
                        mel_spec = DataProcessing.get_mel_spectrogram_tensorflow(audio)
                        plt.imshow(mel_spec)
                        plt.colorbar(label="dB")
                        plt.title("Mel-Spectrogramme Log")
                        plt.xlabel("Temps")
                        plt.ylabel("Echelle MEL")
                        plt.show()
                        ct+=1
                plt.show()


    """
    @staticmethod
    def select_dataset_part(dataset, class_name):
        filtered_audio = []
        filtered_labels = []

        # Liste des indices de classes disponibles
        class_index = dataset.class_names.index(class_name)  # obtenir l'index de la classe "class_name"
        print(class_index)
        random_indices = np.random.choice(10300, 1000, replace=False)

        idx_set = set(random_indices)  # convertir en set pour rechercher plus rapidement

        # Itération sur les batches du dataset
        for batch_idx, (audio, label) in enumerate(dataset):
            for i in range(len(audio)):  # On itère sur chaque élément du batch
                # Vérifier si l'étiquette de l'élément est de la classe voulue
                if label[i] == class_index:
                    global_index = batch_idx * 64 + i  # Indice global de l'élément
                    # Si l'élément est dans les indices aléatoires, on l'ajoute
                    if global_index in idx_set:
                        filtered_audio.append(audio[i])
                        filtered_labels.append(label[i])
                else:
                    # Ajouter tous les éléments de l'autre classe
                    filtered_audio.append(audio[i])
                    filtered_labels.append(label[i])

        return filtered_audio, filtered_labels
        """
    
    def select_dataset_part(self,class_name,number=9000):
        answer = input("ARE YOU SURE YOU WANT TO DELETE FILES ?")
        if answer:
            path_class = os.path.join(self.dataset_dir,class_name)

            files = [f for f in os.listdir(path_class)]
            print(f"Size file before {len(files)}")

            file_to_remove = random.sample(files,number)
            for file_name in file_to_remove:
                path_to_remove = os.path.join(path_class,file_name)
                os.remove(path_to_remove)
            files_after = [f for f in os.listdir(path_class)]
            print(f"Size file after {len(files_after)}")
        else:
            print("No FILES REMOVED")


    
    def get_spectrogram_dataset(self):
        print("GO")
        train_dataset = tf.keras.utils.audio_dataset_from_directory(
            directory=self.dataset_dir,
            batch_size=64,
            validation_split=0.05,  # 20% des données iront en validation
            subset="training",  # Partie training
            seed=42,
            output_sequence_length=16000
        )

        val_dataset = tf.keras.utils.audio_dataset_from_directory(
            directory=self.dataset_dir,
            batch_size=64,
            validation_split=0.05,  # 20% des données iront en validation
            subset="validation",  # Partie validation
            seed=42,
            output_sequence_length=16000
        )

        print(train_dataset.element_spec)
        label_names = np.array(train_dataset.class_names)
        print("Label names:", label_names)

        # ⚡️ Appliquer `squeeze()` pour enlever les dimensions inutiles
        train_dataset = train_dataset.map(DataProcessing.squeeze, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.map(DataProcessing.squeeze, num_parallel_calls=tf.data.AUTOTUNE)

        # ⚡️ Transformer en spectrogramme
        train_dataset = train_dataset.map(lambda audio, label: (DataProcessing.get_spectrogram(audio), label),
                                          num_parallel_calls=tf.data.AUTOTUNE)

        val_dataset = val_dataset.map(lambda audio, label: (DataProcessing.get_spectrogram(audio), label),
                                      num_parallel_calls=tf.data.AUTOTUNE)

        print(train_dataset.element_spec)

        # 🎯 On veut maintenant un dataset de test. Prenons 10% des données de train
        test_size = int(0.30 * sum(1 for _ in train_dataset))  # 75% du validation set

        # On crée un dataset de test avec `take()` et on réduit train avec `skip()`
        test_dataset = train_dataset.take(test_size)
        train_dataset = train_dataset.skip(test_size)

        return train_dataset, val_dataset, test_dataset
    

    def get_mel_spectrogram_dataset(self):
        print("GO")
        train_dataset = tf.keras.utils.audio_dataset_from_directory(
            directory=self.dataset_dir,
            batch_size=64,
            validation_split=0.05,  # 20% des données iront en validation
            subset="training",  # Partie training
            seed=42,
            output_sequence_length=16000
        )

        val_dataset = tf.keras.utils.audio_dataset_from_directory(
            directory=self.dataset_dir,
            batch_size=64,
            validation_split=0.05,  # 20% des données iront en validation
            subset="validation",  # Partie validation
            seed=42,
            output_sequence_length=16000
        )

        print(train_dataset.element_spec)
        label_names = np.array(train_dataset.class_names)
        print("Label names:", label_names)

        # ⚡️ Appliquer `squeeze()` pour enlever les dimensions inutiles
        train_dataset = train_dataset.map(DataProcessing.squeeze, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.map(DataProcessing.squeeze, num_parallel_calls=tf.data.AUTOTUNE)

        # ⚡️ Transformer en spectrogramme
        train_dataset = train_dataset.map(lambda audio, label: (DataProcessing.get_mel_spectrogram_tensorflow(audio), label),
                                          num_parallel_calls=tf.data.AUTOTUNE)

        val_dataset = val_dataset.map(lambda audio, label: (DataProcessing.get_mel_spectrogram_tensorflow(audio), label),
                                      num_parallel_calls=tf.data.AUTOTUNE)

        print(train_dataset.element_spec)

        # 🎯 On veut maintenant un dataset de test. Prenons 10% des données de train
        #test_size = int(0.30 * sum(1 for _ in train_dataset))  # 75% du validation set

        test_size = int(0.30 * tf.data.experimental.cardinality(train_dataset).numpy())  

        # On crée un dataset de test avec `take()` et on réduit train avec `skip()`
        test_dataset = train_dataset.take(test_size)
        train_dataset = train_dataset.skip(test_size)

        train_dataset = train_dataset.cache().shuffle(2000).prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.cache().prefetch(tf.data.AUTOTUNE)
        test_dataset = test_dataset.cache().prefetch(tf.data.AUTOTUNE)

        return train_dataset, val_dataset, test_dataset
    



    @staticmethod
    def get_wigner_ville_distribution(waveform):
        """
        Calcule la distribution de Wigner-Ville pour un waveform donné.
        """
        def _compute_wigner_ville(waveform_np):
            analytic_signal = signal.hilbert(waveform_np)
            tfr = signal.spectrogram(analytic_signal, fs=16000, window='hann', nperseg=256, noverlap=128, mode='complex')
            f, t, Sxx = tfr
            WV = np.abs(Sxx)**2
            WV = np.log(WV + 1e-6)  # Pour éviter log(0)
            WV -= WV.min()
            WV /= WV.max()
            return WV.astype(np.float32)

        wv_distribution = tf.py_function(func=_compute_wigner_ville, inp=[waveform], Tout=tf.float32)
        wv_distribution.set_shape((129, 124))  # Ajuster la forme selon vos besoins
        return wv_distribution[..., tf.newaxis]
    


    @staticmethod
    def wigner_distribution(x, use_analytic=True, sample_frequency=None,
                            t_0=0, t_1=1, flip_frequency_range=True):
        """Discrete Pseudo Wigner Ville Distribution based on [1]

        Args:
            x, array like, signal input array of length N
            use_analytic, bool, whether or not to use analytic associate of input
                data x by default set to True
            sample_frequency, sampling frequency
            t_0, time at which the first sample was recorded
            t_1, time at which the last sample was recorded
            flip_frequency_range, flip the data in about the time axis such that
                the minimum frequency is in the left bottom corner.

        Returns:
            wigner_distribution, N x N matrix
            max_frequency, a positive number, maximum of the frequency range

        References:
            [1] T. Claasen & W. Mecklenbraeuker, The Wigner Distribution -- A Tool
            For Time-Frequency Signal Analysis, Phillips J. Res. 35, 276-300, 1980
        """

        # Ensure the input array is a numpy array
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)
        # Compute the autocorrelation function matrix
        if x.ndim != 1:
            raise ValueError("Input data should be one dimensional time series.")
        # Use analytic associate if set to True
        if use_analytic:
            if all(np.isreal(x)):
                x = signal.hilbert(x)
            else:
                raise RuntimeError("Keyword 'use_analytic' set to True but signal"
                                " is of complex data type. The analytic signal"
                                " can only be computed if the input signal is"
                                " real valued.")

        # calculate the wigner distribution
        N = x.shape[0]
        bins = np.arange(N)
        print("BINS:",bins)
        indices = linalg.hankel(bins, bins + N - (N % 2))
        print("INDICES:",indices)
        padded_x = np.pad(x, (N, N), 'constant')
        print("PADDED_X:",padded_x)
        wigner_integrand = \
            padded_x[indices+N] * np.conjugate(padded_x[indices[::, ::-1]])

        wigner_distribution = np.real(np.fft.fft(wigner_integrand, axis=1)).T
        print("WIGNER_DISTRIBUTION:",wigner_distribution)

        # calculate sample frequency
        if sample_frequency is None:
            sample_frequency = N / (t_1 - t_0)

        # calculate frequency range
        if use_analytic:
            max_frequency = sample_frequency/2
        else:
            max_frequency = sample_frequency/4

        # flip the frequency range
        if flip_frequency_range:
            wigner_distribution = wigner_distribution[::-1, ::]

        return wigner_distribution, max_frequency
    


    def plot_Wigner_Ville(self,n,type="yes_drone"):
        for class_name in os.listdir(self.dataset_dir):
            if class_name==type:
                class_path = os.path.join(self.dataset_dir,class_name)
                ct = 0
                plt.figure(figsize=(16,10))
                for file_name in os.listdir(class_path):
                    while ct <n:
                        audio_path = os.path.join(class_path,file_name)
                        sample_rate, audio = wavfile.read(audio_path)
                        #audio = audio.astype(np.float32)
                        WV_spectro = DataProcessing.wigner_distribution(audio)[0]
                        plt.imshow(WV_spectro)
                        plt.colorbar(label="dB")
                        plt.title("Wigner-Ville Spectrogramme")
                        plt.xlabel("Temps")
                        plt.ylabel("Frequence")
                        plt.gca().invert_yaxis()
                        plt.show()
                        ct+=1
                plt.show()
    


    def get_wigner_ville_dataset(self):
        print("GO")
        train_dataset = tf.keras.utils.audio_dataset_from_directory(
            directory=self.dataset_dir,
            batch_size=64,
            validation_split=0.05,
            subset="training",
            seed=42,
            output_sequence_length=16000
        )

        val_dataset = tf.keras.utils.audio_dataset_from_directory(
            directory=self.dataset_dir,
            batch_size=64,
            validation_split=0.05,
            subset="validation",
            seed=42,
            output_sequence_length=16000
        )

        print(train_dataset.element_spec)
        label_names = np.array(train_dataset.class_names)
        print("Label names:", label_names)

        # Appliquer `squeeze()` pour enlever les dimensions inutiles
        train_dataset = train_dataset.map(DataProcessing.squeeze, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.map(DataProcessing.squeeze, num_parallel_calls=tf.data.AUTOTUNE)

        # Transformer en distribution de Wigner-Ville
        train_dataset = train_dataset.map(lambda audio, label: (DataProcessing.get_wigner_ville_distribution(audio), label),
                                          num_parallel_calls=tf.data.AUTOTUNE)

        val_dataset = val_dataset.map(lambda audio, label: (DataProcessing.get_wigner_ville_distribution(audio), label),
                                      num_parallel_calls=tf.data.AUTOTUNE)

        print(train_dataset.element_spec)

        # Créer un dataset de test
        test_size = int(0.30 * tf.data.experimental.cardinality(train_dataset).numpy())
        test_dataset = train_dataset.take(test_size)
        train_dataset = train_dataset.skip(test_size)

        train_dataset = train_dataset.cache().shuffle(2000).prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.cache().prefetch(tf.data.AUTOTUNE)
        test_dataset = test_dataset.cache().prefetch(tf.data.AUTOTUNE)

        return train_dataset, val_dataset, test_dataset
    
    
    
    @staticmethod
    def get_mfcc(waveform):
        stfts = tf.signal.stft(waveform, frame_length=256, frame_step=128,fft_length=1024)
        spectrograms = tf.abs(stfts)
        
        # Conversion en échelle Mel
        num_spectrogram_bins = spectrograms.shape[-1]
        lower_edge_hertz, upper_edge_hertz = 0, 10000
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins = 80, num_spectrogram_bins=spectrograms.shape[-1], sample_rate=16000, lower_edge_hertz = 0 , upper_edge_hertz= 1000)
        mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
        
        # Log-magnitude Mel spectrogram
        log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
        
        # Extraction des MFCCs
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)
        return mfccs[..., tf.newaxis]
    
    def plot_mfcc(self,n,type="yes_drone"):
        for class_name in os.listdir(self.dataset_dir):
            if class_name==type:
                class_path = os.path.join(self.dataset_dir,class_name)
                ct = 0
                plt.figure(figsize=(16,10))
                for file_name in os.listdir(class_path):
                    while ct <n:
                        audio_path = os.path.join(class_path,file_name)
                        sample_rate, audio = wavfile.read(audio_path)
                        audio = audio.astype(np.float32)
                        mfcc = DataProcessing.get_mfcc(audio)
                        plt.imshow(mfcc)
                        plt.colorbar(label="dB")
                        plt.title("MFCC")
                        plt.xlabel("Temps")
                        plt.ylabel("MFCC")
                        plt.show()
                        ct+=1
                plt.show()
                
                
    def get_mfcc_dataset(self):
        print("GO")
        train_dataset = tf.keras.utils.audio_dataset_from_directory(
            directory=self.dataset_dir,
            batch_size=64,
            validation_split=0.05,  # 20% des données iront en validation
            subset="training",  # Partie training
            seed=42,
            output_sequence_length=16000
        )

        val_dataset = tf.keras.utils.audio_dataset_from_directory(
            directory=self.dataset_dir,
            batch_size=64,
            validation_split=0.05,  # 20% des données iront en validation
            subset="validation",  # Partie validation
            seed=42,
            output_sequence_length=16000
        )

        print(train_dataset.element_spec)
        label_names = np.array(train_dataset.class_names)
        print("Label names:", label_names)

        # ⚡️ Appliquer `squeeze()` pour enlever les dimensions inutiles
        train_dataset = train_dataset.map(DataProcessing.squeeze, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.map(DataProcessing.squeeze, num_parallel_calls=tf.data.AUTOTUNE)

        # ⚡️ Transformer en spectrogramme
        train_dataset = train_dataset.map(lambda audio, label: (DataProcessing.get_mfcc(audio), label),
                                          num_parallel_calls=tf.data.AUTOTUNE)

        val_dataset = val_dataset.map(lambda audio, label: (DataProcessing.get_mfcc(audio), label),
                                      num_parallel_calls=tf.data.AUTOTUNE)

        print(train_dataset.element_spec)

        # 🎯 On veut maintenant un dataset de test. Prenons 10% des données de train
        #test_size = int(0.30 * sum(1 for _ in train_dataset))  # 75% du validation set

        test_size = int(0.30 * tf.data.experimental.cardinality(train_dataset).numpy())  

        # On crée un dataset de test avec `take()` et on réduit train avec `skip()`
        test_dataset = train_dataset.take(test_size)
        train_dataset = train_dataset.skip(test_size)

        train_dataset = train_dataset.cache().shuffle(2000).prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.cache().prefetch(tf.data.AUTOTUNE)
        test_dataset = test_dataset.cache().prefetch(tf.data.AUTOTUNE)

        return train_dataset, val_dataset, test_dataset
        
    @staticmethod
    def bin_frequency(drone):

        membo_fft = tf.signal.fft(tf.cast(drone, tf.complex64))  # FFT en TensorFlow
        membo_fft = tf.concat([[membo_fft[0]], 2 * membo_fft[1:]], axis=0)  
        P_membo_fft = tf.abs(membo_fft)  # Puissance spectrale

        f_fft_membo = tf.signal.fftfreq(16000, d=1/16000)  # Fréquences de la FFT

        fb = tf.range(0, 8000, 10, dtype=tf.float32)  # Bins de fréquence

        # 📌 4. Regroupement par bins
        bin_indices = tf.searchsorted(fb, f_fft_membo, side='right') - 1 
        bin_indices = tf.clip_by_value(bin_indices, 0, len(fb) - 1) 
        Mean_energy_bin = tf.math.unsorted_segment_sum(P_membo_fft, bin_indices, num_segments=len(fb))
        number_el_bin = tf.math.unsorted_segment_sum(tf.ones_like(P_membo_fft), bin_indices, num_segments=len(fb))
        number_el_bin = tf.where(number_el_bin == 0, tf.ones_like(number_el_bin), number_el_bin)
        Mean_energy_bin = Mean_energy_bin / number_el_bin
        return Mean_energy_bin
    
    
    def get_bin_frequency_dataset(self):
        print("GO")
        train_dataset = tf.keras.utils.audio_dataset_from_directory(
            directory=self.dataset_dir,
            batch_size=64,
            validation_split=0.05,  # 20% des données iront en validation
            subset="training",  # Partie training
            seed=42,
            output_sequence_length=16000
        )

        val_dataset = tf.keras.utils.audio_dataset_from_directory(
            directory=self.dataset_dir,
            batch_size=64,
            validation_split=0.05,  # 20% des données iront en validation
            subset="validation",  # Partie validation
            seed=42,
            output_sequence_length=16000
        )

        print(train_dataset.element_spec)
        label_names = np.array(train_dataset.class_names)
        print("Label names:", label_names)

        # récupérer les données de bin de fréquence
        train_dataset = train_dataset.map(lambda audio, label: (DataProcessing.bin_frequency(audio), label),
                                          num_parallel_calls=tf.data.AUTOTUNE)

        val_dataset = val_dataset.map(lambda audio, label: (DataProcessing.bin_frequency(audio), label),
                                      num_parallel_calls=tf.data.AUTOTUNE)

        print(train_dataset.element_spec)

        # 🎯 On veut maintenant un dataset de test. Prenons 10% des données de train
        #test_size = int(0.30 * sum(1 for _ in train_dataset))  # 75% du validation set

        test_size = int(0.30 * tf.data.experimental.cardinality(train_dataset).numpy())  

        # On crée un dataset de test avec `take()` et on réduit train avec `skip()`
        test_dataset = train_dataset.take(test_size)
        train_dataset = train_dataset.skip(test_size)

        train_dataset = train_dataset.cache().shuffle(2000).prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.cache().prefetch(tf.data.AUTOTUNE)
        test_dataset = test_dataset.cache().prefetch(tf.data.AUTOTUNE)

        return train_dataset, val_dataset, test_dataset
        
    
    
    