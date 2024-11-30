import matplotlib.pyplot as plt
import numpy as np

def przebiegi(nazwa_czujnika, atrybut_docelowy):
    probkowania_slownik = {
        "PS1": 100, "PS2": 100, "PS3": 100, "PS4": 100, "PS5": 100, "PS6": 100, "EPS1": 100, "FS1": 10, "FS2": 10, "TS1": 1, "TS2": 1, "TS3": 1, "TS4": 1, "VS1": 1, "CE": 1, "CP": 1, "SE": 1
    }
    
    parametryPrzykaldowe = [
        {"cooler condition": 0, "reduced efficiency": 726, "full efficiency": 1470}, 
        {"optimal switching behavior": 1, "small lag": 238, "severe lag": 226, "close to total failure": 216}, 
        {"no leakage": 2, "weak leakage": 286, "severe leakage": 222}, 
        {"optimal pressure": 3, "slightly reduced pressure": 335, "severely reduced pressure": 509, "close to total failure": 644}, 
        {"stable conditions": 246, "static conditions might not have been reached yet": 23}
    ]
    
    parametry = parametryPrzykaldowe[atrybut_docelowy]
    probkowanie = probkowania_slownik[nazwa_czujnika]
    print(f"Sampling frequency for {nazwa_czujnika}: {probkowanie} Hz")
    
    probki_dict = {}  
    with open(f"{nazwa_czujnika}.txt", 'r') as file:
        lines = [line.split() for line in file if line.strip()]  
    
    for klucz, numer_linii in parametry.items():
        if numer_linii < len(lines):
            probki_dict[klucz] = [float(x) for x in lines[numer_linii]]  
        else:
            probki_dict[klucz] = []  
  
    for klucz, probki in probki_dict.items():
        if probki:
            czas = [i / probkowanie for i in range(len(probki))]  
            
            plt.figure(figsize=(10, 6))
            plt.subplot(2, 1, 1)
            plt.plot(czas, probki)
            plt.title(f"{klucz} - {nazwa_czujnika}")
            plt.xlabel("Time [s]")
            plt.ylabel("Value")
            plt.grid()

            N = len(probki)
            T = 1.0 / probkowanie
            xf = np.fft.fftfreq(N, T)  
            yf = np.fft.fft(probki)    

            pos_idx = np.where(xf >= 0)  
            xf_pos = xf[pos_idx]
            yf_pos = yf[pos_idx]

            plt.subplot(2, 1, 2)
            plt.plot(xf_pos, np.abs(yf_pos))  # Wyświetlamy amplitudę tylko dla dodatnich częstotliwości
            plt.title(f"Frequency spectrum of {klucz} - {nazwa_czujnika}")
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Amplitude")
            plt.grid()

            plt.tight_layout()  
            plt.show()
        else:
            print(f"No data for key: {klucz}")
    
    return probki_dict

print(przebiegi("PS3", 3))
