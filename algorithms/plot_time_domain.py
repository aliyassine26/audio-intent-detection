import librosa
import matplotlib.pyplot as plt

def plot_time_domain(data,rate):

    plt.plot(librosa.samples_to_time(range(len(data)), sr=rate), data)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Waveform')
    plt.show()      
    