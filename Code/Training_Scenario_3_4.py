# -*- coding: utf-8 -*-


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
#import keras as K
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import gc


phys = "../././././Preprocessed_Var_Scenario_3/scenario_3"
annot = ".././././././scenario_3"

print(os.path.exists(phys))

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

'''
common = list(set(all_list) & set(phys_list) & set(emg_list))

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



def iteration(number, train_x, train_y, valid_x, valid_y, log_file, modal, fold):
    tf.keras.backend.clear_session()
    gc.collect()
    
    model = create_model()
    # Define callback to save the best validation model
    checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model_'+ str(number)+'_'+fold+ '_'+modal+'.h5', monitor='val_loss', save_best_only=True)
    # Define callback to log mse score of train and validation set
    mse_logger = tf.keras.callbacks.CSVLogger(log_file, separator=',', append=True)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    model.fit(train_x, train_y, epochs=750, batch_size=20, verbose=1, validation_data=(valid_x, valid_y), callbacks=[checkpoint,mse_logger, early_stopping])
    # Load the best validation model
    model.load_weights('best_model_'+ str(number)+'_'+fold+ '_'+modal+'.h5')
    # Evaluate the model on train and validation set and log the mse score in a text file
    with open(log_file, 'a') as f:
        f.write(f'\n\Subject: {number}\n')
        f.write(f'\n\Fold: {fold}\n')
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
subs =  ['21', '22', '23', '26', '30', '32', '33', '34', '35', '36']
for root, dirs, files in os.walk(phys):
    for file in files: 
        
        print(file)
        x = file.split('.')[0].split('_')[1]
        y = file.split('.')[0].split('_')[-1]
        if not x in subs: continue
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

for i in data: 
    print(i)
    for j in range(4):
        print(j)
        test_fold = "fold_"+str(j)
        t1 = "fold_"+str((j+1)%4) 
        t2 = "fold_"+str((j+2)%4)
        t3 = "fold_"+str((j+3)%4)
        
        print("test")
        test_ff = [test_fold]
        print(test_fold)
        print("train")
        print(t1)
        print(t2)
        print(t3)
        train_ff = [t1,t2,t3]
        print("  ")
        train_x_all = []
        train_x_phys = []
        train_x_emg = []
        train_y = []
        valid_x_all = []
        valid_x_phys = []
        valid_x_emg = []
        valid_y = []
        for k in data[i]:
            print(k)
            for h in data[i][k]:
                if h == "test": continue
                for g in data[i][k][h]:
                    print(g)
                    df_train = pd.read_csv(data[i][k][h][g]["physiology"])
                    df_valid = pd.read_csv(data[i][k][h][g]["annotations"])[['valence','arousal']]
                    df_valid = df_valid[-(len(df_valid)-1):]
                    
                    if k in train_ff:
                        train_y = train_y + df_valid.values.tolist()
                        grouped = df_train.groupby(df_train.index // 50)
                    
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
                    
                    else:
                        valid_y = valid_y + df_valid.values.tolist()
                        grouped = df_train.groupby(df_train.index // 50)
                    
                        flattened = grouped.apply(lambda x: x['all_sig'].values.flatten().tolist())
                        result = flattened.tolist()
                        to_app = result[:-1]
                        to_app = check_list(to_app, j)
                        valid_x_all = valid_x_all +to_app
                    
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

        if not i in all_list:     iteration(i, np.asarray(train_x_all).astype(float), np.asarray(train_y).astype(float), np.asarray(valid_x_all).astype(float), np.asarray(valid_y).astype(float), "log_all.txt", "all", test_fold)
        if not i in phys_list:   iteration(i, np.asarray(train_x_phys).astype(float), np.asarray(train_y).astype(float), np.asarray(valid_x_phys).astype(float), np.asarray(valid_y).astype(float), "log_phys.txt", "phys", test_fold)
        if not i in emg_list:  iteration(i, np.asarray(train_x_emg).astype(float), np.asarray(train_y).astype(float), np.asarray(valid_x_emg).astype(float), np.asarray(valid_y).astype(float), "log_emg.txt", "emg", test_fold)
                    
            
        
        
