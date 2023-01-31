import librosa
import numpy as np
import pandas as pd

def pad_audio(data,sample_rate,target_length):
    # Pad audio file to target length
    data_pad = librosa.util.fix_length(data, size=target_length * sample_rate)

    temp_series = pd.Series(dtype=float)
    temp_series.loc["data_pad"] = data_pad
    temp_series.loc["duration_data"] = librosa.get_duration(y=data_pad, sr=sample_rate)

    return temp_series