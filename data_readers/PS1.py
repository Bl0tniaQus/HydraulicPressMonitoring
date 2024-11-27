import numpy as np
import scipy.signal
from scipy.stats import kurtosis, skew
import matplotlib.pyplot as plt
import helper_functions
fs = 0 #Hz
data = np.loadtxt("../raw_data/PS1.txt")
print(np.shape(data))
row = data[0]
duration = 60  # s
means = np.apply_along_axis(np.mean, 1, data)
mins = np.apply_along_axis(np.min, 1, data)
maxes = np.apply_along_axis(np.max, 1, data)
RMS = np.apply_along_axis(helper_functions.rms, 1, data)
krt = np.apply_along_axis(kurtosis,1,data)
sd = np.apply_along_axis(np.std, 1, data)
skw = np.apply_along_axis(skew, 1, data)
p2p = maxes - mins
#momenty centralne, dyspersja i pewnie coś jeszcze
#t = np.linspace(0, duration, int(fs * duration), endpoint=False)
#fft_result = np.fft.fft(row)
#freq = np.fft.fftfreq(t.shape[-1], d=1/fs)

plt.plot(range(len(row)), row)
plt.show()
