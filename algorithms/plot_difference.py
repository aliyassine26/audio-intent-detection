import librosa
import matplotlib.pyplot as plt
import noisereduce as nr

def plot_difference(data,sample_rate):

    trimmed_data, index = librosa.effects.trim(data, top_db=20, frame_length=512, hop_length=256)
    data_clean = nr.reduce_noise(y=trimmed_data, sr=sample_rate)

    plt.plot(librosa.samples_to_time(range(len(data))),data, label="Original Wave",alpha=0.7)
    plt.plot(librosa.samples_to_time(range(len(data_clean))),data_clean, label="Modified Wave",alpha=0.7)
    plt.legend()
    plt.show()
    librosa.samples_to_time(range(len(data)))