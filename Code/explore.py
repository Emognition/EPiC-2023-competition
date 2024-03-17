# -*- coding: utf-8 -*-

# Import necessary libraries
import numpy as np
import pandas as pd
import os
import sys 
import matplotlib.pyplot as plt

# Define paths for physiological and annotation data
phys = r'D:\Codes\Personal\2023\ACII\EPiC Challenge\scenario_1\train\physiology'
annot = r"D:\Codes\Personal\2023\ACII\EPiC Challenge\scenario_1\train\annotations"

# Define a dictionary to store the data
data = {}
data["Subject"] = []
data["Video"] = []

# Define a list of physiological signal names
name_phy = ["ecg","bvp","gsr","rsp","skt","emg_zygo","emg_coru","emg_trap"]

# Add the physiological signal names as keys to the data dictionary
for i in name_phy:
    data[i] = []

# Initialize some variables
cor = 0
num = 0 
var = 0
num2 = 0

# Walk through the directories to read the data files and merge them
for root, dirs, files in os.walk(phys):
    for file in files:
        # Print the name of the file being processed
        print(file)
        # Extract the subject and video numbers from the file name
        fn = os.path.splitext(file)[0].split('_')

        # Read the physiological and annotation data files and merge them on the 'time' column
        phys_df = pd.read_csv(os.path.join(root, file))
        ann_df = pd.read_csv(os.path.join(annot, file))
        merged_df = phys_df.merge(ann_df, on='time', how="outer")
        # Interpolate the missing values
        merged_df.interpolate(method='linear', inplace=True)
        # Check if there is any missing correlation value
        if not merged_df.corr().isna().any().any():
            cor = cor + merged_df.corr()
            num += 1
        # Calculate the variance and count of the data samples
        var = var + merged_df.var()
        num2 = num2+1
        #sys.exit(1)

# Define a list of colors for plotting the signals
colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta', 'black', 'gray']

# Walk through the directories again to plot the physiological signals
for root, dirs, files in os.walk(phys):
    for file in files:
        # Print the name of the file being processed
        print(file)
        # Read the physiological data file and convert time from ms to s
        df = pd.read_csv(os.path.join(root, file))
        df['time'] = df['time'] / 1000
        
        # Plot each physiological signal in a separate figure
        for i, col in enumerate(df.columns[1:]):
            plt.figure()
            plt.plot(df['time'], df[col], color=colors[i])
            plt.xlabel('Time (s)')
            plt.ylabel("amplitude")
            plt.title(col)
            
            plt.show()