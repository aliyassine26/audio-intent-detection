import librosa
import librosa.display
import matplotlib.pyplot as plt
import sklearn

def plot_mfcc(path):
    data ,rate = librosa.load(path)
    mfcc = librosa.feature.mfcc(y=data, sr=rate, n_mfcc=50)
    mfcc = sklearn.preprocessing.scale(mfcc, axis=1)
    librosa.display.specshow(mfcc, sr=rate, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()