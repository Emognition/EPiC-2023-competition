# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 15:06:04 2023

@author: ntweat
"""
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
import csv
import math
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Input
from sklearn import preprocessing
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score, confusion_matrix
from statistics import mean 
import pandas as pd
from scipy.signal import resample
import keras as K
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import gc
from sklearn.ensemble import RandomForestClassifier


phys = r"F:\Codes\Personal\2023\ACII\Preprocessed_Var_Scenario_3\scenario_3"
annot = r"D:\Codes\Personal\2023\ACII\EPiC Challenge\scenario_3"

store_path = r"D:\Codes\Personal\2023\ACII\scenario_3_results"
def create_model():
    opt = tf.keras.optimizers.Adadelta(lr=0.01)
    input_layer = tf.keras.layers.Input((51,))
    dense1 = tf.keras.layers.Dense(units=2500, activation='relu')(input_layer)
    dense2 = tf.keras.layers.Dense(units=500, activation='relu')(dense1)
    dense3 = tf.keras.layers.Dense(units=250, activation='relu')(dense2)
    output_layer = tf.keras.layers.Dense(units=2, activation='linear')(dense3)
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
    return model


def check_list(my_list, k):
    for i in range(len(my_list)): 
        sub_list = my_list[i]
        if len(sub_list) < 50:
            num_zeros_to_append = 50 - len(sub_list)
            my_list[i].extend([0] * num_zeros_to_append)
        my_list[i].append(k)
        
    return my_list
        
data = {}

for root, dirs, files in os.walk(phys):
    for file in files: 
        
        print(file)
        x = file.split('.')[0].split('_')[1]
        y = file.split('.')[0].split('_')[-1]
        subdirs = root.split(os.sep)
        fp1 = subdirs[6]
        fp2 = subdirs[7]
        fp3 = subdirs[8]
        fp4 = subdirs[9]
        ann_file = os.path.join(annot, fp2, fp3, "annotations", file)
        print(ann_file)
        if not x in data:
            data[x] = {}
        if not fp2 in data[x]:
            data[x][fp2] = {}
        if not fp3 in data[x][fp2]:
            data[x][fp2][fp3] ={}
        if not fp4 in data[x][fp2][fp3]:
            data[x][fp2][fp3][y] = {}
        # in data[x][fp2][fp3][y]:
        data[x][fp2][fp3][y][fp4] = os.path.join(root, file)
        data[x][fp2][fp3][y]["annotations"] = ann_file


def predict (number, test_x, modal, fold):
    tf.keras.backend.clear_session()
    gc.collect()
    
    model = create_model()
    model.load_weights('best_model_'+ str(number)+'_'+fold+ '_'+modal+'.h5')
    preds = model.predict(test_x)
    return preds



for i in data: 
    print(i)
    for j in range(4):
        print(j)
        test_fold = "fold_"+str(j)
        t1 = "fold_"+str((j+1)%2) 
        #t2 = "fold_"+str((j+2)%4)
        #t3 = "fold_"+str((j+3)%4)
        
        print("test")
        test_ff = [t1, test_fold]
        print(test_fold)
        print("train")
        print(t1)
        #print(t2)
        #print(t3)
        train_ff = [t1, test_fold]
        print("  ")
        test_x_all = []
        test_x_phys = []
        test_x_emg = []
        train_y = []
        valid_x_all = []
        valid_x_phys = []
        valid_x_emg = []
        valid_y = []
        for k in data[i]:
            print(k)
            for h in data[i][k]:
                if not h == "test": continue
                print(h)
                for g in data[i][k][h]:
                    print(g)
                    df_train = pd.read_csv(data[i][k][h][g]["physiology"])
                    df_valid = pd.read_csv(data[i][k][h][g]["annotations"])[['valence','arousal']]
                    an_df = pd.read_csv(data[i][k][h][g]["annotations"])
                    XX = len(an_df)
                    fname = data[i][k][h][g]["annotations"]
                    file = os.path.basename(fname)
                    grouped = df_train.groupby(df_train.index // 50)
                    
                    flattened = grouped.apply(lambda x: x['all_sig'].values.flatten().tolist())
                    result = flattened.tolist()
                    to_app = result
                    test_x_all = check_list(to_app, j)
                    
                    
                    flattened = grouped.apply(lambda x: x['phys_sig'].values.flatten().tolist())
                    result = flattened.tolist()
                    to_app = result
                    test_x_phys = check_list(to_app, j)
                    
                    
                    flattened = grouped.apply(lambda x: x['emg_sig'].values.flatten().tolist())
                    result = flattened.tolist()
                    to_app = result
                    test_x_emg = check_list(to_app, j)
                    
                    
                    res = predict(i, np.asarray(test_x_all).astype(float), "all", test_fold)
                    
                    res_path = os.path.join(store_path, "all",  h, k)
                    
                    if not os.path.exists(res_path):
                        os.makedirs(res_path)
                    
                    
                    
                    
                    last_X_row = res[-XX:]
                    an_df[['valence', 'arousal']] = last_X_row
                    
                    an_df.to_csv(os.path.join(res_path, file), index=False)
                    
                    
                    res = predict(i, np.asarray(test_x_phys).astype(float), "phys", test_fold)
                    
                    res_path = os.path.join(store_path, "phys",  h, k)
                    
                    if not os.path.exists(res_path):
                        os.makedirs(res_path)
                    
                    
                    
                    
                    last_X_row = res[-XX:]
                    an_df[['valence', 'arousal']] = last_X_row
                    
                    an_df.to_csv(os.path.join(res_path, file), index=False)
                    
                    
                    res = predict(i, np.asarray(test_x_emg).astype(float), "emg", test_fold)
                    
                    res_path = os.path.join(store_path, "emg",  h, k)
                    
                    if not os.path.exists(res_path):
                        os.makedirs(res_path)
                    
                    
                    
                    
                    last_X_row = res[-XX:]
                    an_df[['valence', 'arousal']] = last_X_row
                    
                    an_df.to_csv(os.path.join(res_path, file), index=False)
                    
                    
                    
                    #sys.exit(1)
