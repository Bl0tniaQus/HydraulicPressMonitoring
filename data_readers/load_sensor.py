import numpy as np
import scipy.signal
from scipy.stats import kurtosis, skew, moment
import matplotlib.pyplot as plt
import helper_functions
import pandas as pd

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

def load_sensor(dataframe, sensor_name):
    fs = sampling_rates[sensor_name]
    data = np.loadtxt("../raw_data/"+sensor_name+".txt")
    means = np.apply_along_axis(np.mean, 1, data)
    mins = np.apply_along_axis(np.min, 1, data)
    maxes = np.apply_along_axis(np.max, 1, data)
    p2p = maxes - mins
    RMS = np.apply_along_axis(helper_functions.rms, 1, data)
    #expected_value = np.apply_along_axis(moment, 1, data, 1)
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
    #dataframe[sensor_name+"_expected_value"] = expected_value
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
        dataframe[sensor_name+f"_f({freq[0]}-{freq[1]})_mean"] = np.apply_along_axis(helper_functions.mean_f, 1, data, fs, freq[0], freq[1])
        dataframe[sensor_name+f"_f({freq[0]}-{freq[1]})_max"] = np.apply_along_axis(helper_functions.max_f, 1, data, fs, freq[0], freq[1])
        dataframe[sensor_name+f"_f({freq[0]}-{freq[1]})_sd"] = np.apply_along_axis(helper_functions.sd_f, 1, data, fs, freq[0], freq[1])
        dataframe[sensor_name+f"_f({freq[0]}-{freq[1]})_variance"] = np.apply_along_axis(helper_functions.variance_f, 1, data, fs, freq[0], freq[1])
        dataframe[sensor_name+f"_f({freq[0]}-{freq[1]})_RMS"] = np.apply_along_axis(helper_functions.rms_f, 1, data, fs, freq[0], freq[1])
        dataframe[sensor_name+f"_f({freq[0]}-{freq[1]})_skewiness"] = np.apply_along_axis(helper_functions.skew_f, 1, data, fs, freq[0], freq[1])
        dataframe[sensor_name+f"_f({freq[0]}-{freq[1]})_kurtosis"] = np.apply_along_axis(helper_functions.kurtosis_f, 1, data, fs, freq[0], freq[1])


    return dataframe
#TODO Frequency domain features, some more time domain features
#t = np.linspace(0, duration, int(fs * duration), endpoint=False)
#fft_result = np.fft.fft(row)
#freq = np.fft.fftfreq(t.shape[-1], d=1/fs)

