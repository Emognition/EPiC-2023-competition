# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 12:47:33 2023

@author: ntweat
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 00:27:39 2023

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

phys = r"D:\Codes\Personal\2023\ACII\EPiC Challenge\scenario_1\train\physiology"
annot = r"D:\Codes\Personal\2023\ACII\EPiC Challenge\scenario_1\train\annotations"

test_phys = r"D:\Codes\Personal\2023\ACII\EPiC Challenge\scenario_1\test\physiology"
test_annot = r"D:\Codes\Personal\2023\ACII\EPiC Challenge\scenario_1\test\annotations"

store_results = r"D:\Codes\Personal\2023\ACII\scenario_1_results"
def read_txt(filename):
    subject_lines = []  # Create an empty list to store the subject lines

    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('\Subject:'):
                subject_lines.append(line.strip().split()[-1])  # Remove any leading or trailing whitespace

    return subject_lines

all_list = []
phys_list =[]
emg_list =[]
'''
all_list = read_txt('log_all.txt')
phys_list = read_txt('log_phys.txt')

emg_list = read_txt('log_emg.txt')

#'''
common = list(set(all_list) & set(phys_list) & set(emg_list))


    
#'''
def create_model():
    opt = tf.keras.optimizers.Adadelta(lr=0.1)
    input_layer = tf.keras.layers.Input((51,))
    dense1 = tf.keras.layers.Dense(units=2500, activation='relu')(input_layer)
    dense2 = tf.keras.layers.Dense(units=500, activation='relu')(dense1)
    dense3 = tf.keras.layers.Dense(units=250, activation='relu')(dense2)
    output_layer = tf.keras.layers.Dense(units=2, activation='linear')(dense3)
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
    return model
'''

def create_model():
    opt = tf.keras.optimizers.Adadelta(lr=0.01)
    input_layer = tf.keras.layers.Input((51,))
    dense1 = tf.keras.layers.Dense(units=25, activation='relu')(input_layer)
    #dense2 = tf.keras.layers.Dense(units=500, activation='relu')(dense1)
    #dense3 = tf.keras.layers.Dense(units=250, activation='relu')(dense2)
    output_layer = tf.keras.layers.Dense(units=2, activation='linear')(dense1)
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
    return model


def iteration(number, train_x, train_y, valid_x, valid_y, log_file, modal):
    tf.keras.backend.clear_session()
    gc.collect()
    
    model = create_model()
    # Define callback to save the best validation model
    checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model_'+ str(number)+'_'+modal+'.h5', monitor='val_loss', save_best_only=True)
    # Define callback to log mse score of train and validation set
    mse_logger = tf.keras.callbacks.CSVLogger(log_file, separator=',', append=True)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    model.fit(train_x, train_y, epochs=750, batch_size=20, verbose=1, validation_data=(valid_x, valid_y), callbacks=[checkpoint,mse_logger, early_stopping])
    # Load the best validation model
    model.load_weights('best_model_'+ str(number)+'_'+modal+'.h5')
    # Evaluate the model on train and validation set and log the mse score in a text file
    with open(log_file, 'a') as f:
        f.write(f'\n\Subject: {number}\n')
        train_mse = model.evaluate(train_x, train_y, verbose=0)[1]
        valid_mse = model.evaluate(valid_x, valid_y, verbose=0)[1]
        f.write(f'Train MSE: {train_mse:.4f}\n')
        f.write(f'Valid MSE: {valid_mse:.4f}\n')
        f.write('-' * 50)
      
'''
def predict (number, test_x, modal):
    tf.keras.backend.clear_session()
    gc.collect()
    
    model = create_model()
    model.load_weights('best_model_'+ str(number)+'_'+modal+'.h5')
    preds = model.predict(test_x)
    return preds

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
        fn = os.path.splitext(file)[0].split('_')
        if not fn[1] in data:
            data[fn[1]] ={}
        if not fn[-1] in data[fn[1]]:
            data[fn[1]][fn[-1]] = {}
            
        phys_df = pd.read_csv(os.path.join(root, file))
        ann_df = pd.read_csv(os.path.join(annot, file))
        
        
        data[fn[1]][fn[-1]]["train_X"] = phys_df
        data[fn[1]][fn[-1]]["train_Y"] = ann_df
        

varrs = {}
for i in data:
    varrs[i] = {}
    print(i)
    var_df = pd.DataFrame()
    
    for j in data[i]:
        
        var_df = pd.concat([var_df, data[i][j]["train_X"]])
        
    varr = var_df.var()
    data[i]["var"] = varr
    
    data[i]["scale"] = MinMaxScaler()
    data[i]["scale"].fit(var_df)
    
    varrs[i]["var"] = varr 
    varrs[i]["scale"] = MinMaxScaler()
    varrs[i]["scale"].fit(var_df)
    
    


del data 
all_col = ['ecg','bvp','gsr','rsp','skt','emg_zygo','emg_coru','emg_trap']
phys_col = ['ecg','bvp','gsr','rsp','skt']
emg_Col = ['emg_zygo','emg_coru','emg_trap']


for root, dirs, files in os.walk(test_phys):
    for file in files:
        fn = os.path.splitext(file)[0].split('_')
        subs = fn[1]
        tas = fn[-1]
        test_x_all = []
        test_x_phys = []
        test_x_emg = []
        
        an_df = pd.read_csv(os.path.join(test_annot, file))
        
        XX = len(an_df)
        
        test_df = pd.read_csv(os.path.join(root, file))
        var_df = test_df.copy()
        var_df.values[:,:] = varrs[i]['scale'].transform(test_df)[:,:]
        var_df = var_df*varrs[i]['var']
        var_df['time'] = test_df['time']
        var_df['all_sig'] = var_df[all_col].sum(axis=1)
        var_df['phys_sig'] = var_df[phys_col].sum(axis=1)
        var_df['emg_sig'] = var_df[emg_Col].sum(axis=1)
        
        grouped = var_df.groupby(var_df.index // 50)
        flattened = grouped.apply(lambda x: x['all_sig'].values.flatten().tolist())
        result = flattened.tolist()
        to_app = result[:-1]
        to_app = check_list(to_app, tas)
        test_x_all = test_x_all +to_app
        
        flattened = grouped.apply(lambda x: x['phys_sig'].values.flatten().tolist())
        result = flattened.tolist()
        to_app = result[:-1]
        to_app = check_list(to_app, tas)
        test_x_phys = test_x_phys +to_app
        
        flattened = grouped.apply(lambda x: x['emg_sig'].values.flatten().tolist())
        result = flattened.tolist()
        to_app = result[:-1]
        to_app = check_list(to_app, tas)
        test_x_emg = test_x_emg +to_app
        
        res = predict(subs, np.asarray(test_x_all).astype(float), "all")
        last_X_row = res[-XX:]
        an_df[['valence', 'arousal']] = last_X_row
        
        all_test_path = os.path.join(store_results, "all")
        if not os.path.exists(all_test_path):
            os.makedirs(all_test_path)
            
        an_df.to_csv(os.path.join(all_test_path, file), index=False)
        
        
        res = predict(subs, np.asarray(test_x_all).astype(float), "emg")
        last_X_row = res[-XX:]
        an_df[['valence', 'arousal']] = last_X_row
        
        all_test_path = os.path.join(store_results, "emg")
        if not os.path.exists(all_test_path):
            os.makedirs(all_test_path)
            
        an_df.to_csv(os.path.join(all_test_path, file), index=False)
        
        
        res = predict(subs, np.asarray(test_x_all).astype(float), "phys")
        last_X_row = res[-XX:]
        an_df[['valence', 'arousal']] = last_X_row
        
        all_test_path = os.path.join(store_results, "phys")
        if not os.path.exists(all_test_path):
            os.makedirs(all_test_path)
            
        an_df.to_csv(os.path.join(all_test_path, file), index=False)
        
        #sys.exit(1)
        

