import matplotlib.pyplot as plt 
import numpy as np 

x_array = np.linspace(-90,90,10000)
y_array = np.sin(x_array)*np.log10(x_array)

plt.figure()
plt.plot(x_array,y_array,label="ground truth")
plt.xlabel("angles")
plt.ylabel("Mag")
plt.title("Mag evolution")
plt.grid(True)
plt.legend()
plt.show()

from data_processing import DataProcessing

file = "/home/raphalinux/PycharmProjects/pythonProject/projet_sys/data/Binary_Drone_Audio"
dataset_binary = DataProcessing(file)
dataset_binary.plot_waveform(4)
dataset_binary.plot_waveform(2,"unknown")
dataset_binary.plot_spectogram(1)