import load_sensor
import pandas as pd
sensors = ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "TS1", "TS2", "TS3", "TS4", "FS1", "FS2", "SE", "VS1", "CP", "CE", "EPS1"]

frame = pd.DataFrame()

for sensor in sensors:
    frame = load_sensor.load_sensor(frame, sensor)
frame.to_csv("../data.csv")
