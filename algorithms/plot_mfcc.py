import librosa
import librosa.display
import matplotlib as plt

def plot_mfcc(path):
    data ,rate = librosa.load(path)
    mfcc = librosa.feature.mfcc(y=data, sr=rate, n_mfcc=50)
    librosa.display.specshow(mfcc, sr=rate, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()