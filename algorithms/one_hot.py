def one_hot_encode(dataframe,cols):
    return pd.get_dummies(dataframe, columns=[cols])