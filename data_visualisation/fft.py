import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt("../raw_data/PS1.txt")
signal = data[0]
n = signal.shape[-1]
print(n) #6000
fs = 100
T = 1 / fs
x_t = np.array(range(n)) * T
y_t = signal
plt.figure(figsize=(5.5, 6))
plt.subplot(2, 1, 1)
plt.plot(x_t, y_t)
plt.title(f"PS1(t)")
plt.xlabel("Time [s]")
plt.ylabel("Value")
plt.grid()
x_f = np.fft.fftfreq(n, T)
y_f = np.fft.fft(signal)
f_filter = np.where(x_f > 0)
x_f = x_f[f_filter]
y_f = y_f[f_filter]
plt.subplot(2, 1, 2)
plt.plot(x_f, np.abs(y_f))
plt.title(f"PS1(f)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")
plt.grid()
plt.tight_layout()
plt.show()
print(x_f[1] - x_f[0]) #0.0167

