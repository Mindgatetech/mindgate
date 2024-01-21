from sklearn.preprocessing import MinMaxScaler

def my_scaler():
    scaler = MinMaxScaler()
    return ('Scaler', scaler)