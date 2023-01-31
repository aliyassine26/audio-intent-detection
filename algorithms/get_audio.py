import librosa
import numpy as np
import pandas as pd

def get_audio(path):
    data, sample_rate = librosa.core.load(path)

    temp_series = pd.Series(dtype=float)
    temp_series.loc["data"] = data
    temp_series.loc["sample_rate"] = sample_rate
    temp_series.loc["duration"] = librosa.get_duration(y=data, sr=sample_rate)



    return temp_series