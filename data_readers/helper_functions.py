import numpy as np

def rms(row):
    return np.divide(np.sqrt(np.sum(np.square(row))), np.sqrt(row.shape[0]))
