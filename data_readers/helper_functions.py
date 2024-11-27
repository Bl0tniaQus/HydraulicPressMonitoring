import numpy as np

def rms(row):
    return np.sqrt(np.square(row)) / np.sqrt(row.shape[0])
