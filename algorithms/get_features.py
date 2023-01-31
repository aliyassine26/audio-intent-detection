import librosa
import numpy as np
import pandas as pd
from scipy.stats import skew

def get_features(path):
    data, sample_rate = librosa.core.load(path)

    # Trim the audio file
    trimmed_data, index = librosa.effects.trim(data, top_db=20, frame_length=512, hop_length=256)
    mfcc = librosa.feature.mfcc(trimmed_data, sr = sample_rate, n_mfcc=30)
    zcr = librosa.feature.zero_crossing_rate(trimmed_data)[0]
    rolloff = librosa.feature.spectral_rolloff(trimmed_data)[0]
    spectral_centroid = librosa.feature.spectral_centroid(trimmed_data)[0]
    spectral_contrast = librosa.feature.spectral_contrast(trimmed_data)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(trimmed_data)[0]
    full_duration = librosa.get_duration(y=data, sr=sample_rate)
    trimed_duration = librosa.get_duration(y=trimmed_data, sr=sample_rate)

    # Compute STFT & magnitude spectrogram
    # stft = librosa.stft(data)
    # spectrogram = np.abs(stft)

    # spectrogram_array = np.hstack((np.mean(spectrogram, axis=1), np.std(spectrogram, axis=1), skew(spectrogram, axis = 1), np.max(spectrogram, axis = 1), np.median(spectrogram, axis = 1), np.min(spectrogram, axis = 1)))
    mfcc_array = np.hstack((np.mean(mfcc, axis=1), np.std(mfcc, axis=1), skew(mfcc, axis = 1), np.max(mfcc, axis = 1), np.median(mfcc, axis = 1), np.min(mfcc, axis = 1)))
    zcr_array = np.hstack((np.mean(zcr), np.std(zcr), skew(zcr), np.max(zcr), np.median(zcr), np.min(zcr)))
    rolloff_array = np.hstack((np.mean(rolloff), np.std(rolloff), skew(rolloff), np.max(rolloff), np.median(rolloff), np.min(rolloff)))
    spectral_centroid_array = np.hstack((np.mean(spectral_centroid), np.std(spectral_centroid), skew(spectral_centroid), np.max(spectral_centroid), np.median(spectral_centroid), np.min(spectral_centroid)))
    spectral_contrast_array = np.hstack((np.mean(spectral_contrast), np.std(spectral_contrast), skew(spectral_contrast), np.max(spectral_contrast), np.median(spectral_contrast), np.min(spectral_contrast)))
    spectral_bandwidth_array = np.hstack((np.mean(spectral_bandwidth), np.std(spectral_bandwidth), skew(spectral_bandwidth), np.max(spectral_bandwidth), np.median(spectral_bandwidth), np.max(spectral_bandwidth)))
    
    return pd.Series(np.hstack((mfcc_array, zcr_array, rolloff_array, spectral_centroid_array, spectral_contrast_array, spectral_bandwidth_array,full_duration,trimed_duration)))