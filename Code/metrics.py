# -*- coding: utf-8 -*-


import pandas as pd
import math

# Define the path to the log files
log_all_path = r'D:\Codes\Personal\2023\ACII\SC4\log_all.txt'
log_phys_path = r'D:\Codes\Personal\2023\ACII\SC4\log_phys.txt'
log_emg_path = r'D:\Codes\Personal\2023\ACII\SC4\log_emg.txt'

# Define a dictionary to map filenames to abbreviations
file_abbrevs = {
    log_all_path: 'All',
    log_phys_path: 'Phys',
    log_emg_path: 'Emg'
}

# Define an empty list to store the extracted data
data = []

# Loop through each log file and extract the relevant data
for log_path in [log_all_path, log_phys_path, log_emg_path]:
    with open(log_path, 'r') as f:
        for line in f:
            if 'Subject' in line:
                subject = int(line.split()[1])
            elif 'Train MSE' in line:
                train_mse = math.sqrt(float(line.split()[2]))
            elif 'Valid MSE' in line:
                valid_mse = math.sqrt(float(line.split()[2]))
                data.append([subject, train_mse, valid_mse, file_abbrevs[log_path]])

# Convert the extracted data to a Pandas DataFrame
df = pd.DataFrame(data, columns=['Subject', 'Train MSE', 'Valid MSE', 'File'])

# Calculate the mean and standard deviation of the Train MSE and Valid MSE columns
train_mse_mean = df['Train MSE'].mean()
train_mse_std = df['Train MSE'].std()
valid_mse_mean = df['Valid MSE'].mean()
valid_mse_std = df['Valid MSE'].std()

print(f'Train MSE Mean: {train_mse_mean:.2f}')
print(f'Train MSE Std Dev: {train_mse_std:.2f}')
print(f'Valid MSE Mean: {valid_mse_mean:.2f}')
print(f'Valid MSE Std Dev: {valid_mse_std:.2f}')




for i in df['File'].unique():
    dg = df[df['File']==i]
    train_mse_mean = dg['Train MSE'].mean()
    train_mse_std = dg['Train MSE'].std()
    valid_mse_mean = dg['Valid MSE'].mean()
    valid_mse_std = dg['Valid MSE'].std() 
    print(f'Modality: {i}')
    print(f'Train MSE Mean: {train_mse_mean:.2f}')
    print(f'Train MSE Std Dev: {train_mse_std:.2f}')
    print(f'Valid MSE Mean: {valid_mse_mean:.2f}')
    print(f'Valid MSE Std Dev: {valid_mse_std:.2f}')
