import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
fs = 0 #Hz
data = np.loadtxt("../raw_data/PS1.txt")
print(np.shape(data))
row = data[0]
duration = 60  # s
#t = np.linspace(0, duration, int(fs * duration), endpoint=False)
frequency = 440  # A4 note frequency (Hz)

#fft_result = np.fft.fft(row)
#freq = np.fft.fftfreq(t.shape[-1], d=1/fs)

plt.plot(range(len(row)), row)
plt.show()
