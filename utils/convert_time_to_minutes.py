"""Convert Time column to minutes elapsed since first timestamp."""

import pandas as pd
from pathlib import Path

csv_file = "homogenized_sensor_data.csv"
df = pd.read_csv(csv_file)
df['Time'] = pd.to_datetime(df['Time'])
start_time = df['Time'].iloc[0]
time_minutes = (df['Time'] - start_time).dt.total_seconds() / 60
df['Time'] = time_minutes.astype(int)  # 5 min intervals
df.to_csv(csv_file, index=False)

