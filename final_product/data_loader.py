import pandas as pd
import numpy as np
import scipy.signal
import os
from scipy.stats import kurtosis, skew, moment
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
sensors = ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "TS1", "TS2", "TS3", "TS4", "FS1", "FS2", "SE", "VS1", "CP", "CE", "EPS1"]
frequency_ranges = {
"PS1": [(0,1)],
"PS2": [(0,7)],
"PS3": [(0,20), (20, 50)],
"PS4": [(0,20), (20, 50)],
"PS5": [(0, 0.1)],
"PS6": [(0,0.1)],
"TS1": [(0,0.07)],
"TS2": [(0,0.07)],
"TS3": [(0,0.07)],
"TS4": [(0,0.07)],
"FS1": [(0,1), (1, 5)],
"FS2": [(0,0.1)],
"SE": [(0,0.1), (0.1, 0.5)],
"VS1": [(0,0.1), (0.1, 0.3)],
"CP": [(0, 0.1), (0.1,0.3)],
"CE": [(0,0.1), (0.1, 0.5)],
"EPS1": [(0,1)]
}
sampling_rates = {
        "PS1": 100, "PS2": 100, "PS3": 100, "PS4": 100, "PS5": 100, "PS6": 100, "EPS1": 100, "FS1": 10, "FS2": 10, "TS1": 1, "TS2": 1, "TS3": 1, "TS4": 1, "VS1": 1, "CE": 1, "CP": 1, "SE": 1
    }
def loadAll(data_dir):
    frame = pd.DataFrame()
    target = pd.read_csv(os.path.join(data_dir, "profile.txt"), sep="\t", header=None, names=["cooler", "valve", "pump", "bar", "stable"])
    for sensor in sensors:
        frame = load_sensor(frame, sensor, data_dir)
    frame.fillna(0, inplace=True)
    frame = pd.concat([frame, target], axis=1)
    print(frame.columns.to_series()[np.isinf(frame).any()])
    frame.to_csv("data.csv", index=False)
def load_sensor(dataframe, sensor_name, data_dir):
    fs = sampling_rates[sensor_name]
    data = np.loadtxt(os.path.join(data_dir, f"{sensor_name}.txt"))
    means = np.apply_along_axis(np.mean, 1, data)
    mins = np.apply_along_axis(np.min, 1, data)
    maxes = np.apply_along_axis(np.max, 1, data)
    p2p = maxes - mins
    RMS = np.apply_along_axis(rms, 1, data)
    variance = np.apply_along_axis(moment, 1, data, 2)
    sd = np.sqrt(variance)
    moment3 = np.apply_along_axis(moment, 1, data, 3)
    moment4 = np.apply_along_axis(moment, 1, data, 4)
    moment5 = np.apply_along_axis(moment, 1, data, 5)
    moment6 = np.apply_along_axis(moment, 1, data, 6)
    moment7 = np.apply_along_axis(moment, 1, data, 7)
    moment8 = np.apply_along_axis(moment, 1, data, 8)
    moment9 = np.apply_along_axis(moment, 1, data, 9)
    moment10 = np.apply_along_axis(moment, 1, data, 10)
    krt = np.apply_along_axis(kurtosis,1,data)
    np.nan_to_num(krt)
    skw = np.apply_along_axis(skew, 1, data)
    np.nan_to_num(skw)
    variation = np.divide(sd, means)
    dataframe[sensor_name+"_mean"] = means
    dataframe[sensor_name+"_min"] = mins
    dataframe[sensor_name+"_maxes"] = maxes
    dataframe[sensor_name+"_peak2peak"] = p2p
    dataframe[sensor_name+"_RMS"] = RMS
    dataframe[sensor_name+"_variance"] = variance
    dataframe[sensor_name+"_sd"] = sd
    dataframe[sensor_name+"_3rd_moment"] = moment3
    dataframe[sensor_name+"_4th_moment"] = moment4
    dataframe[sensor_name+"_5th_moment"] = moment5
    dataframe[sensor_name+"_6th_moment"] = moment6
    dataframe[sensor_name+"_7th_moment"] = moment7
    dataframe[sensor_name+"_8th_moment"] = moment8
    dataframe[sensor_name+"_9th_moment"] = moment9
    dataframe[sensor_name+"_10th_moment"] = moment10
    dataframe[sensor_name+"_kurtosis"] = krt
    dataframe[sensor_name+"_skeweness"] = skw
    dataframe[sensor_name+"_variation"] = variation

    for freq in frequency_ranges[sensor_name]:
        dataframe[sensor_name+f"_f({freq[0]}-{freq[1]})_mean"] = np.apply_along_axis(mean_f, 1, data, fs, freq[0], freq[1])
        dataframe[sensor_name+f"_f({freq[0]}-{freq[1]})_max"] = np.apply_along_axis(max_f, 1, data, fs, freq[0], freq[1])
        dataframe[sensor_name+f"_f({freq[0]}-{freq[1]})_sd"] = np.apply_along_axis(sd_f, 1, data, fs, freq[0], freq[1])
        dataframe[sensor_name+f"_f({freq[0]}-{freq[1]})_variance"] = np.apply_along_axis(variance_f, 1, data, fs, freq[0], freq[1])
        dataframe[sensor_name+f"_f({freq[0]}-{freq[1]})_RMS"] = np.apply_along_axis(rms_f, 1, data, fs, freq[0], freq[1])
        dataframe[sensor_name+f"_f({freq[0]}-{freq[1]})_skewiness"] = np.apply_along_axis(skew_f, 1, data, fs, freq[0], freq[1])
        dataframe[sensor_name+f"_f({freq[0]}-{freq[1]})_kurtosis"] = np.apply_along_axis(kurtosis_f, 1, data, fs, freq[0], freq[1])
        dataframe[sensor_name+f"_f({freq[0]}-{freq[1]})_variation"] = np.apply_along_axis(variation_f, 1, data, fs, freq[0], freq[1])
    
    return dataframe
    
def loadObservation(observation_dict):
    values = []
    for sensor in sensors:
        values = getFeaturesForObservation(values, observation_dict[sensor], sensor)
    values = np.array(values)
    values = np.nan_to_num(values, nan = 0)
    return values
def getFeaturesForObservation(values, measurements, sensor_name):
    if sensor_name not in sensors:
        raise ValueError("Invalid input data")
    fs = sampling_rates[sensor_name]
    data = measurements
    means = np.mean(data)
    mins = np.min(data)
    maxes = np.max(data)
    p2p = maxes - mins
    RMS = rms(data)
    variance = moment(data,2)
    sd = np.sqrt(variance)
    moment3 = moment(data,3)
    moment4 = moment(data,4)
    moment5 = moment(data,5)
    moment6 = moment(data,6)
    moment7 = moment(data,7)
    moment8 = moment(data,8)
    moment9 = moment(data,9)
    moment10 = moment(data,10)
    krt = kurtosis(data)
    skw = skew(data)
    variation = np.divide(sd, means)
    values.append(means)
    values.append(mins)
    values.append(maxes)
    values.append(p2p)
    values.append(RMS)
    values.append(variance)
    values.append(sd)
    values.append(moment3)
    values.append(moment4)
    values.append(moment5)
    values.append(moment6)
    values.append(moment7)
    values.append(moment8)
    values.append(moment9)
    values.append(moment10)
    values.append(krt)
    values.append(skw)
    values.append(variation)
    for freq in frequency_ranges[sensor_name]:
        values.append(mean_f(data, fs, freq[0], freq[1]))
        values.append(max_f(data, fs, freq[0], freq[1]))
        values.append(sd_f(data, fs, freq[0], freq[1]))
        values.append(variance_f(data, fs, freq[0], freq[1]))
        values.append(rms_f(data, fs, freq[0], freq[1]))
        values.append(skew_f(data, fs, freq[0], freq[1]).item())
        values.append(kurtosis_f(data, fs, freq[0], freq[1]).item())
        values.append(variation_f(data, fs, freq[0], freq[1]))
    return values
def rms(row):
    return np.divide(np.sqrt(np.sum(np.square(row))), np.sqrt(row.shape[0]))
def rms_f(row, fs, fmin, fmax):
    f = np.fft.fftfreq(row.shape[0], 1/fs)
    v = np.fft.fft(row)
    f_range = np.argwhere((f >= fmin) & (f <= fmax) & (f > 0))
    values = np.abs(v[f_range])
    values = np.nan_to_num(values)
    return np.divide(np.sqrt(np.sum(np.square(values))), np.sqrt(values.shape[0]))

def max_f(row, fs, fmin, fmax):
    f = np.fft.fftfreq(row.shape[0], 1/fs)
    v = np.fft.fft(row)
    f_range = np.argwhere((f >= fmin) & (f <= fmax) & (f > 0))
    values = np.abs(v[f_range])
    values = np.nan_to_num(values)
    return np.max(values)

def mean_f(row, fs, fmin, fmax):
    f = np.fft.fftfreq(row.shape[0], 1/fs)
    v = np.fft.fft(row)
    f_range = np.argwhere((f >= fmin) & (f <= fmax) & (f > 0))
    values = np.abs(v[f_range])
    values = np.nan_to_num(values)
    return np.mean(values)

def sd_f(row, fs, fmin, fmax):
    f = np.fft.fftfreq(row.shape[0], 1/fs)
    v = np.fft.fft(row)
    f_range = np.argwhere((f >= fmin) & (f <= fmax) & (f > 0))
    values = np.abs(v[f_range])
    values = np.nan_to_num(values)
    return np.std(values)

def skew_f(row, fs, fmin, fmax):
    f = np.fft.fftfreq(row.shape[0], 1/fs)
    v = np.fft.fft(row)
    f_range = np.argwhere((f >= fmin) & (f <= fmax) & (f > 0))
    values = np.abs(v[f_range])
    values = np.nan_to_num(values)
    return np.nan_to_num(skew(values))

def kurtosis_f(row, fs, fmin, fmax):
    f = np.fft.fftfreq(row.shape[0], 1/fs)
    v = np.fft.fft(row)
    f_range = np.argwhere((f >= fmin) & (f <= fmax) & (f > 0))
    values = np.abs(v[f_range])
    values = np.nan_to_num(values)
    return np.nan_to_num(kurtosis(values))

def variance_f(row, fs, fmin, fmax):
    f = np.fft.fftfreq(row.shape[0], 1/fs)
    v = np.fft.fft(row)
    f_range = np.argwhere((f >= fmin) & (f <= fmax) & (f > 0))
    values = np.abs(v[f_range])
    values = np.nan_to_num(values)
    return np.var(values)

def variation_f(row, fs, fmin, fmax):
    f = np.fft.fftfreq(row.shape[0], 1/fs)
    v = np.fft.fft(row)
    f_range = np.argwhere((f >= fmin) & (f <= fmax) & (f > 0))
    values = np.abs(v[f_range])
    values = np.nan_to_num(values)
    return np.divide(sd_f(row, fs, fmin, fmax), mean_f(row, fs, fmin, fmax))
    
def readDictFromFiles(data_dir, idx):
    result = {}
    for sensor in sensors:
        data = np.loadtxt(os.path.join(data_dir, f"{sensor}.txt"))
        result[sensor] = data[idx]
    return result
    
