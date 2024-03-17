# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 23:32:32 2023

@author: ntweat
"""

import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import sys

folder_path = r"D:\Codes\Personal\2023\ACII\EPiC Challenge\scenario_4"

output_path = r'D:\Codes\Personal\2023\ACII\Preprocessed_Scenario_4'

'''
subdirs = root.split(os.sep)
fp1 = subdirs[6]
fp2 = subdirs[7]
fp3 = subdirs[8]
fp4 = subdirs[9]
sys.exit(1)
print(root)
print(file)
'''

subject_groups = {}

for root, dirs, files in os.walk(folder_path):
    for file in files:
        if root.endswith('annotations'): 
            #print('yes')
            continue
        print(file)
        x = file.split('.')[0].split('_')[1]
        y = file.split('.')[0].split('_')[-1]
        if x not in subject_groups:
            subject_groups[x] = MinMaxScaler()
        
        df = pd.read_csv(os.path.join(root, file))
        subject_groups[x].partial_fit(df)
            
        #sys.exit(1)
        
        

if not os.path.exists(output_path):
    os.makedirs(output_path)
    
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if root.endswith('annotations'): 
            #print('yes')
            continue
        print(file)
        x = file.split('.')[0].split('_')[1]
        y = file.split('.')[0].split('_')[-1]
        
        subdirs = root.split(os.sep)
        fp1 = subdirs[6]
        fp2 = subdirs[7]
        fp3 = subdirs[8]
        fp4 = subdirs[9]
        
        df = pd.read_csv(os.path.join(root, file))
        
        df_scaled = pd.DataFrame(subject_groups[x].transform(df), columns=df.columns)
        
        df_scaled['time'] = df['time']
        
        out_folder = os.path.join(output_path, fp1, fp2, fp3, fp4)
        
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        
        output_file = os.path.join(out_folder, file)
        df_scaled.to_csv(output_file, index=False)

        
        
