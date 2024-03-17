# -*- coding: utf-8 -*-


# Import required libraries
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

# Define the paths to the data files
phys = r"D:\Codes\Personal\2023\ACII\processed_data_scenario_1_train"
annot = r"D:\Codes\Personal\2023\ACII\EPiC Challenge\scenario_1\train\annotations"

# Define a function to read in data from text files
def read_txt(filename):
    subject_lines = []  # Create an empty list to store the subject lines

    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('\Subject:'):
                subject_lines.append(line.strip().split()[-1])  # Remove any leading or trailing whitespace

    return subject_lines

# Initialize some variables for storing data
all_list = []
phys_list =[]
emg_list =[]

# Find the common elements in the three lists
common = list(set(all_list) & set(phys_list) & set(emg_list))

# Define a function to create the neural network model
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

# Define a function for each iteration of the training process
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
        

for i in data:
    print(i)
    var_df = pd.DataFrame()
    
    for j in data[i]:
        
        var_df = pd.concat([var_df, data[i][j]["train_X"]])
        
    varr = var_df.var()
    data[i]["var"] = varr
    scc = MinMaxScaler()
    data[i]["scale"] = scc.fit(var_df)
    


all_col = ['ecg','bvp','gsr','rsp','skt','emg_zygo','emg_coru','emg_trap']
phys_col = ['ecg','bvp','gsr','rsp','skt']
emg_Col = ['emg_zygo','emg_coru','emg_trap']


for i in data:
    print(i)
    for j in data[i]:
        if j == "var": continue
        var_df = (data[i][j]['train_X']*data[i]['scale'])*data[i]['var']
        var_df['time'] = data[i][j]['train_X']['time']
        var_df['all_sig'] = var_df[all_col].sum(axis=1)
        var_df['phys_sig'] = var_df[phys_col].sum(axis=1)
        var_df['emg_sig'] = var_df[emg_Col].sum(axis=1)
        data[i][j]['train_X'] = var_df
        
        
def convert_to_float(my_list):
    for i in range(len(my_list)):
        for j in range(len(my_list[i])):
            my_list[i][j] = float(my_list[i][j])
    return my_list

for i in data:
    print(i)
    if i in common:
        print("Skipping.....")
        continue
    train_x_all = []
    train_x_phys = []
    train_x_emg = []
    train_y = []
    valid_x_all = []
    valid_x_phys = []
    valid_x_emg = []
    valid_y = []
    for j in data[i]:
        
        if j == 'var': continue
        Y = data[i][j]['train_Y']
        Y = Y[-(len(Y)-1):]
        #sys.exit(1)
        split_index = int(len(Y) * 0.8)
        
        train_y_temp = Y.iloc[:split_index][['valence','arousal']].values.tolist()
        valid_y_temp = Y.iloc[split_index:][['valence','arousal']].values.tolist()
        train_y = train_y + train_y_temp
        valid_y = valid_y + valid_y_temp
        X = data[i][j]['train_X']
        split_time = Y.iloc[:split_index]['time'].iloc[-1]
        df_train_t = X[X['time']<=split_time]
        df_valid_t = X[X['time']>split_time]
        
        grouped = df_train_t.groupby(df_train_t.index // 50)
        flattened = grouped.apply(lambda x: x['all_sig'].values.flatten().tolist())
        result = flattened.tolist()
        to_app = result[:-1]
        to_app = check_list(to_app, j)
        train_x_all = train_x_all +to_app
        
        flattened = grouped.apply(lambda x: x['phys_sig'].values.flatten().tolist())
        result = flattened.tolist()
        to_app = result[:-1]
        to_app = check_list(to_app, j)
        train_x_phys = train_x_phys +to_app
        
        flattened = grouped.apply(lambda x: x['emg_sig'].values.flatten().tolist())
        result = flattened.tolist()
        to_app = result[:-1]
        to_app = check_list(to_app, j)
        train_x_emg = train_x_emg +to_app
        
        
        
        grouped = df_valid_t.groupby(df_valid_t.index // 50)
        flattened = grouped.apply(lambda x: x['all_sig'].values.flatten().tolist())
        result = flattened.tolist()
        to_app = result[:-1]
        to_app = check_list(to_app, j)
        valid_x_all = valid_x_all + to_app
        
        flattened = grouped.apply(lambda x: x['phys_sig'].values.flatten().tolist())
        result = flattened.tolist()
        to_app = result[:-1]
        to_app = check_list(to_app, j)
        valid_x_phys = valid_x_phys +to_app
        
        flattened = grouped.apply(lambda x: x['emg_sig'].values.flatten().tolist())
        result = flattened.tolist()
        to_app = result[:-1]
        to_app = check_list(to_app, j)
        valid_x_emg = valid_x_emg +to_app
        
        
        
        
        
        print( len(train_y_temp) )
        print(len(train_x_emg))
        print(len(train_y))
        
        
        
    if not i in all_list:     iteration(i, np.asarray(train_x_all).astype(float), np.asarray(train_y).astype(float), np.asarray(valid_x_all).astype(float), np.asarray(valid_y).astype(float), "log_all.txt", "all")
    if not i in phys_list:   iteration(i, np.asarray(train_x_phys).astype(float), np.asarray(train_y).astype(float), np.asarray(valid_x_phys).astype(float), np.asarray(valid_y).astype(float), "log_phys.txt", "phys")
    if not i in emg_list:  iteration(i, np.asarray(train_x_emg).astype(float), np.asarray(train_y).astype(float), np.asarray(valid_x_emg).astype(float), np.asarray(valid_y).astype(float), "log_emg.txt", "emg")

        