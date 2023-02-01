import librosa
import pandas as pd

def get_trim(row):
    data = row['data']
    sample_rate = row['sample_rate']

    trimmed_data, index = librosa.effects.trim(data, top_db=20, frame_length=512, hop_length=256)
    duration_trim = librosa.get_duration(y=trimmed_data, sr=sample_rate)

    temp_series = pd.Series(dtype=float)
    temp_series.loc["data_trim"] = trimmed_data
    temp_series.loc["duration_trim"] = duration_trim



    return temp_series