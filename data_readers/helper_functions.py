import numpy as np
from scipy.stats import kurtosis, skew
def rms(row):
    return np.divide(np.sqrt(np.sum(np.square(row))), np.sqrt(row.shape[0]))
def rms_f(row, fs, fmin, fmax):
    f = np.fft.fftfreq(row.shape[0], 1/fs)
    v = np.fft.fft(row)
    f_range = np.argwhere((f >= fmin) & (f <= fmax))
    values = np.abs(v[f_range])
    return np.divide(np.sqrt(np.sum(np.square(values))), np.sqrt(values.shape[0]))

def max_f(row, fs, fmin, fmax):
    f = np.fft.fftfreq(row.shape[0], 1/fs)
    v = np.fft.fft(row)
    f_range = np.argwhere((f >= fmin) & (f <= fmax))
    values = np.abs(v[f_range])
    return np.max(values)

def mean_f(row, fs, fmin, fmax):
    f = np.fft.fftfreq(row.shape[0], 1/fs)
    v = np.fft.fft(row)
    f_range = np.argwhere((f >= fmin) & (f <= fmax))
    values = np.abs(v[f_range])
    return np.mean(values)

def sd_f(row, fs, fmin, fmax):
    f = np.fft.fftfreq(row.shape[0], 1/fs)
    v = np.fft.fft(row)
    f_range = np.argwhere((f >= fmin) & (f <= fmax))
    values = np.abs(v[f_range])
    return np.std(values)

def skew_f(row, fs, fmin, fmax):
    f = np.fft.fftfreq(row.shape[0], 1/fs)
    v = np.fft.fft(row)
    f_range = np.argwhere((f >= fmin) & (f <= fmax))
    values = np.abs(v[f_range])
    return np.nan_to_num(skew(values))

def kurtosis_f(row, fs, fmin, fmax):
    f = np.fft.fftfreq(row.shape[0], 1/fs)
    v = np.fft.fft(row)
    f_range = np.argwhere((f >= fmin) & (f <= fmax))
    values = np.abs(v[f_range])
    return np.nan_to_num(kurtosis(values))

def variance_f(row, fs, fmin, fmax):
    f = np.fft.fftfreq(row.shape[0], 1/fs)
    v = np.fft.fft(row)
    f_range = np.argwhere((f >= fmin) & (f <= fmax))
    values = np.abs(v[f_range])
    return np.var(values)

def variation_f(row, fs, fmin, fmax):
    f = np.fft.fftfreq(row.shape[0], 1/fs)
    v = np.fft.fft(row)
    f_range = np.argwhere((f >= fmin) & (f <= fmax))
    values = np.abs(v[f_range])
    return np.divide(sd_f(row, fs, fmin, fmax), mean_f(row, fs, fmin, fmax))

