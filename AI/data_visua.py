# -*- coding: utf-8 -*-
"""
Created on Sun May  7 16:35:00 2023

@author: COLLINS
"""

import pandas as pd
import matplotlib.pyplot as plt

# Read the dataset from CSV file
df = pd.read_csv('/home/ramanujan/git_repos/BMS-DNN/BMS/dataset/synthetic_soc_data.csv', parse_dates=['Time'])

# Plot the battery's voltage and temperature
fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('Time')
ax1.set_ylabel('Voltage (V)', color=color)
ax1.plot(df['Time'], df['Voltage'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Temperature (°C)', color=color)
ax2.plot(df['Time'], df['Temperature'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.show()
