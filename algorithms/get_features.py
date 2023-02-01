import librosa
import numpy as np
import pandas as pd
from scipy.stats import skew
import sklearn

def get_features(data,sample_rate):

    mfcc = librosa.feature.mfcc(y=data, sr = sample_rate, n_mfcc=30)
    # mfcc = sklearn.preprocessing.scale(mfcc, axis=1)

    spec = librosa.feature.melspectrogram(data, sr=sample_rate, n_mels=128, fmax=8000)

    # use [0] to take first value since below functions return a tuple: (data, frequency)
    zcr = librosa.feature.zero_crossing_rate(y=data)[0]
    rolloff = librosa.feature.spectral_rolloff(y=data)[0]
    spectral_centroid = librosa.feature.spectral_centroid(y=data)[0]
    spectral_contrast = librosa.feature.spectral_contrast(y=data)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=data)[0]

    spec_array = np.hstack((np.mean(spec, axis=1), np.std(spec, axis=1), skew(spec, axis = 1), np.max(spec, axis = 1), np.median(spec, axis = 1), np.min(spec, axis = 1)))
    mfcc_array = np.hstack((np.mean(mfcc, axis=1), np.std(mfcc, axis=1), skew(mfcc, axis = 1), np.max(mfcc, axis = 1), np.median(mfcc, axis = 1), np.min(mfcc, axis = 1)))
    zcr_array = np.hstack((np.mean(zcr), np.std(zcr), skew(zcr), np.max(zcr), np.median(zcr), np.min(zcr)))
    rolloff_array = np.hstack((np.mean(rolloff), np.std(rolloff), skew(rolloff), np.max(rolloff), np.median(rolloff), np.min(rolloff)))
    spectral_centroid_array = np.hstack((np.mean(spectral_centroid), np.std(spectral_centroid), skew(spectral_centroid), np.max(spectral_centroid), np.median(spectral_centroid), np.min(spectral_centroid)))
    spectral_contrast_array = np.hstack((np.mean(spectral_contrast), np.std(spectral_contrast), skew(spectral_contrast), np.max(spectral_contrast), np.median(spectral_contrast), np.min(spectral_contrast)))
    spectral_bandwidth_array = np.hstack((np.mean(spectral_bandwidth), np.std(spectral_bandwidth), skew(spectral_bandwidth), np.max(spectral_bandwidth), np.median(spectral_bandwidth), np.max(spectral_bandwidth)))
    return pd.Series(np.hstack((mfcc_array, zcr_array, rolloff_array, spectral_centroid_array, spectral_contrast_array, spectral_bandwidth_array, spec_array)))