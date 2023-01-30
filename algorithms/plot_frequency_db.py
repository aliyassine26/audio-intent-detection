import librosa
import matplotlib.pyplot as plt



def plot_frequency_db(path):
    # Load the WAV file using librosa
    y, sr = librosa.load(path)


    # Perform a short-time Fourier transform (STFT) to get the frequency data
    D = librosa.stft(y)

    # Convert the amplitude to dB
    D_db = librosa.amplitude_to_db(D)

    #Get the frequencies and times from the STFT
    frequencies = librosa.core.fft_frequencies(sr=sr)
    times = librosa.core.frames_to_time(D.shape[1], sr=sr)

    # Plot the frequency data using a logarithmic y-axis
    plt.plot(frequencies,D_db, color='#1F77B4')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.title('Frequency magnitude (dB)')

    # Show the plot
    plt.show()
