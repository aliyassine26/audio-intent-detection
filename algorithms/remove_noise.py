
import pandas as pd
import noisereduce as nr

def remove_noise(data,sample_rate):
    data_clean = nr.reduce_noise(y=data, sr=sample_rate)

    temp_series = pd.Series(dtype=float)
    temp_series.loc["data_clean"] = data_clean

    return temp_series