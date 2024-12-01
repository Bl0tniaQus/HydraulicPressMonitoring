from plot_graphs import graphs
import pandas as pd
sensors = ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "TS1", "TS2", "TS3", "TS4", "FS1", "FS2", "SE", "VS1", "CP", "CE", "EPS1"]
targets = range(5)
targets = [0]
for sensor in sensors:
    for target in targets:
        graphs(sensor, target)
