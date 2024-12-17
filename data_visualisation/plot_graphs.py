import matplotlib.pyplot as plt
import numpy as np

def graphs(nazwa_czujnika, atrybut_docelowy):
    probkowania_slownik = {
        "PS1": 100, "PS2": 100, "PS3": 100, "PS4": 100, "PS5": 100, "PS6": 100, "EPS1": 100, "FS1": 10, "FS2": 10, "TS1": 1, "TS2": 1, "TS3": 1, "TS4": 1, "VS1": 1, "CE": 1, "CP": 1, "SE": 1
    }
    nazwy_docelowe = ["Cooler condition", "Valve condition", "Internal pump leakage", "Hydraulic accumulator", "Stable?"]
    parametryPrzykaldowe = [
        {"close to total failure": 0, "reduced efficiency": 726, "full efficiency": 1470},
        {"optimal switching behavior": 1, "small lag": 238, "severe lag": 226, "close to total failure": 216}, 
        {"no leakage": 2, "weak leakage": 286, "severe leakage": 222}, 
        {"optimal pressure": 3, "slightly reduced pressure": 335, "severely reduced pressure": 509, "close to total failure": 644}, 
        {"stable conditions": 246, "static conditions might not have been reached yet": 23}
    ]
    
    parametry = parametryPrzykaldowe[atrybut_docelowy]
    probkowanie = probkowania_slownik[nazwa_czujnika]
    nazwa_docelowa = nazwy_docelowe[atrybut_docelowy]
    probki_dict = {}  
    lines = np.loadtxt("../raw_data/"+nazwa_czujnika+".txt")
    n = lines.shape[1]
    czas = [i / probkowanie for i in range(n)]

    for key, value in parametry.items():
        probki = lines[value];
        nazwa = key;
        plt.figure(figsize=(5.5, 6))
        plt.subplot(2, 1, 1)
        plt.plot(czas, probki)
        plt.title(f"{nazwa_czujnika}(t) for {nazwa_docelowa}: {nazwa}")
        plt.xlabel("Time [s]")
        plt.ylabel("Value")
        plt.grid()
        N = len(probki)
        T = 1.0 / probkowanie
        xf = np.fft.fftfreq(N, T)
        yf = np.fft.fft(probki)
        pos_idx = np.where(xf > 0)
        xf_pos = xf[pos_idx]
        yf_pos = yf[pos_idx]
        plt.subplot(2, 1, 2)
        plt.plot(xf_pos, np.abs(yf_pos))  # Wyświetlamy amplitudę tylko dla dodatnich częstotliwości
        plt.title(f"{nazwa_czujnika}(f) for {nazwa_docelowa}: {nazwa}")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.tight_layout()
        plt.show()
    return probki_dict
