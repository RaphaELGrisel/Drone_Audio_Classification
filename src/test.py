import matplotlib.pyplot as plt 
import numpy as np 

from data_processing import DataProcessing
from model import Model

file = "/home/raphalinux/PycharmProjects/pythonProject/projet_sys/data/Binary_Drone_Audio"
dataset_binary = DataProcessing(file)
#dataset_binary.plot_waveform(4)
#dataset_binary.plot_waveform(2,"unknown")
#dataset_binary.plot_spectrogram(1)
#dataset_binary.plot_spectrogram(1,"unknown")
spectro_dataset = dataset_binary.get_spectrogram_dataset()

for example_spectrograms, example_spect_labels in spectro_dataset.take(1):
  break
print(example_spect_labels)
print(example_spectrograms)
print(np.shape(example_spect_labels))
print(np.shape(example_spectrograms))
print(example_spectrograms.shape[1:])
print(spectro_dataset.class_names)

print("#-----------------------------#")

model1 = Model(spectro_dataset)
cnn = model1.CNN((124,129,1),2)
cnn.summary()