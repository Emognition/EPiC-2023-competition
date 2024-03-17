# -*- coding: utf-8 -*-

import pandas as pd 
import sys 
#folder_path = r"D:\Codes\Personal\2023\ACII\EPiC Challenge\scenario_3"

folder_path = r'D:\Codes\Personal\2023\ACII\Preprocessed_Scenario_4'

output_path = r"F:\Codes\Personal\2023\ACII\Preprocessed_Var_Scenario_4"



subject_groups = {}

vared = {}

for root, dirs, files in os.walk(folder_path):
    for file in files:
        if root.endswith('annotations'): 
            continue
        print(file)
        x = file.split('.')[0].split('_')[1]
        y = file.split('.')[0].split('_')[-1]
        if x not in subject_groups:
            subject_groups[x] = pd.DataFrame()
        
        df = pd.read_csv(os.path.join(root, file))
        subject_groups[x] = pd.concat([subject_groups[x], df]) 
        

for i in subject_groups:
    vared[i] = subject_groups[i].var()
    
all_col = ['ecg','bvp','gsr','rsp','skt','emg_zygo','emg_coru','emg_trap']
phys_col = ['ecg','bvp','gsr','rsp','skt']
emg_Col = ['emg_zygo','emg_coru','emg_trap']


if not os.path.exists(output_path):
    os.makedirs(output_path)
    
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if root.endswith('annotations'): 
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
        
        var_df = df*vared[x]
        var_df['all_sig'] = var_df[all_col].sum(axis=1)
        var_df['phys_sig'] = var_df[phys_col].sum(axis=1)
        var_df['emg_sig'] = var_df[emg_Col].sum(axis=1)

        
        var_df['time'] = df['time']
        
        out_folder = os.path.join(output_path, fp1, fp2, fp3, fp4)
        
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        
        output_file = os.path.join(out_folder, file)
        var_df.to_csv(output_file, index=False)
