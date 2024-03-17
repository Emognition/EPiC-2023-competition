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
import pdb

# scenario-2: across individuals
# this code does testing only

# where to put data--it should follow the challenge format. example: REPO_ROOT/data/scenario-2/fold-0/test/annotations and REPO_ROOT/data/scenario-2/fold-0/test/physiology for testing data
root_dir = "../data/scenario_2"

# features given in the challenge
all_features = ['ecg','bvp','gsr','rsp','skt','emg_zygo','emg_coru','emg_trap']

# physiology features in the challenge
phys_features = ['ecg','bvp','gsr','rsp','skt']

# motion features in the challenge
emg_features = ['emg_zygo','emg_coru','emg_trap']

# subset of features to use, can be all_features, phys_features or emg_features. This controls what features are used for training and testing. In final submission, all_features were used.
FEATURES_TO_USE = all_features
# given every 50 samples is annotated. this is used to create individual samples to train the model
ANNOTATION_FREQ = 50
# number of epochs to train
NEPOCHS = 150



def create_model():
    # neural network configuration used for training and testing
    opt = tf.keras.optimizers.Adadelta(lr=0.01)
    input_layer = tf.keras.layers.Input((50,))
    dense1 = tf.keras.layers.Dense(units=2500, activation='relu')(input_layer)
    dense2 = tf.keras.layers.Dense(units=500, activation='relu')(dense1)
    dense3 = tf.keras.layers.Dense(units=250, activation='relu')(dense2)
    output_layer = tf.keras.layers.Dense(units=2, activation='linear')(dense3)
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
    return model


def iteration(nepochs, train_x, train_y, valid_x, valid_y, log_file, modal):
    tf.keras.backend.clear_session()
    gc.collect()
    
    model = create_model()
    
    # Define callback to save the best validation model
    checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(log_file, 'best_model_'+modal+'.h5'), monitor='val_loss', save_best_only=True)
    # Define callback to log mse score of train and validation set
    mse_logger = tf.keras.callbacks.CSVLogger(os.path.join(log_file, modal+'.log'), separator=',', append=True)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, min_delta=0.001, restore_best_weights=True)
    model.fit(train_x, train_y, epochs=nepochs, batch_size=20, verbose=1, validation_data=(valid_x, valid_y), callbacks=[checkpoint, mse_logger, early_stopping])
    # Load the best validation model
    model.load_weights(os.path.join(log_file, 'best_model_'+modal+'.h5'))
    
    # Evaluate the model on train and validation set and log the mse score in a text file
    with open(os.path.join(log_file, modal+'.log'), 'a') as f:
        train_mse = model.evaluate(train_x, train_y, verbose=0)[1]
        valid_mse = model.evaluate(valid_x, valid_y, verbose=0)[1]
        f.write(f'Train MSE: {train_mse:.4f}\n')
        f.write(f'Valid MSE: {valid_mse:.4f}\n')
        f.write('-' * 50)
      


def variance_fusion(fold_partition_data):
  # combine different signals weighted by their variance. This is adopted from Hinduja et al. 2021 where more variance a signal has, more weightage it is given in the resultant signal
  variance = fold_partition_data[FEATURES_TO_USE].var()
  normalized_variance = (variance - variance.min()) / (variance.max() - variance.min())
  fold_partition_data[FEATURES_TO_USE] = fold_partition_data[FEATURES_TO_USE] * normalized_variance
  return fold_partition_data

def compile_data(folder, partition):
  # read each subject-stimulus-setting to preprocess data
  out = list()
  annotations_fold_path = folder.replace('physiology', 'annotations')
  for file in os.listdir(folder):
    # for reference also save meta data, meta data is not used for training
    subject = os.path.basename(file).split('_')[1]
    stimulus = os.path.basename(file).split('_')[3].replace('.csv', '')
    csv_data = pd.read_csv(os.path.join(folder, file))
    # fuse multiple signals into one time-series using variance fusion
    csv_data = variance_fusion(csv_data)
    # convert one timeseries data into multiple row-based samples of length=50 to train model
    csv_data = resample_data_labels(csv_data)
    # z-score normalize data
    csv_data = (csv_data - csv_data.mean()) / csv_data.std() 
    csv_data['subject'] = subject
    csv_data['stimulus'] = stimulus
    if partition == 'train':
        csv_annotations = pd.read_csv(os.path.join(annotations_fold_path, file))
        csv_data['valence'] = csv_annotations['valence']
        csv_data['arousal'] = csv_annotations['arousal']
    out.append(csv_data)

  if isinstance(out, list):
    return pd.concat(out)
  else:
    return out[0]


def get_fold_data(fid, partition='train'):
    # generate training and validation partitions by reading challenge data for each fold
    training_data_list = list()
    validation_data_list = list()
    validation_folds = fid
    if partition == 'train':
      training_folds = {0, 1, 2, 3, 4}.difference({int(fid)})
      for fold in training_folds:
        physiology_fold_path = os.path.join(root_dir, 'fold_{}'.format(fold), partition, 'physiology')
        csv_data = compile_data(physiology_fold_path, partition)
        training_data_list.append(csv_data)

    physiology_fold_path = os.path.join(root_dir, 'fold_{}'.format(fid), partition, 'physiology')
    csv_data = compile_data(physiology_fold_path, partition='train')
    validation_data_list = csv_data
                            
    training_data = pd.concat(training_data_list)
    validation_data = validation_data_list
    if partition == 'train':
      return training_data, validation_data
    else:
      return validation_data
      
def resample_data_labels(input_data):
  # format timeseries data to row-format of length=ANNOTATION_FREQ to be used as input to predict its valence and arousal. No "resampling" is actually done, just a inappropriate name for a function
  fused_signal = input_data[FEATURES_TO_USE].sum(axis=1)
  
  fused_signal_window = list()
  for i in range(0, len(fused_signal)//ANNOTATION_FREQ, 1):
    fused_signal_window.append(fused_signal.iloc[i*ANNOTATION_FREQ: (i+1)*ANNOTATION_FREQ])
  
  resampled_data = pd.DataFrame(np.vstack(fused_signal_window))
  return resampled_data


def test_fold(fid, log_file):
  # use this to generate test predictions that follow the challenge submission format
  features = FEATURES_TO_USE
  folder = os.path.join(root_dir, 'fold_{0}'.format(fid), 'test', 'physiology')
  save_dir = '../results/scenario_2/fold_{0}/test/annotations'.format(fid)
  os.makedirs(save_dir, exist_ok=True)
  out = list()
  annotations_fold_path = folder.replace('physiology', 'annotations')
  # using fold-0 model for test predictions. since we were unable to finish the models for other folds, we use fold-0 models for other folds as well.
  valence_model = create_model()
  valence_model.load_weights(os.path.join('../checkpoints/scenario-2/fold_0', 'best_model_scenario-2_valence.h5'))
  arousal_model = create_model()
  arousal_model.load_weights(os.path.join('../checkpoints/scenario-2/fold_0', 'best_model_scenario-2_arousal.h5'))
 
  for file in os.listdir(folder):
    subject = os.path.basename(file).split('_')[1]
    stimulus = os.path.basename(file).split('_')[3].replace('.csv', '')
    csv_data = pd.read_csv(os.path.join(folder, file))
    annotations = pd.read_csv(os.path.join(annotations_fold_path, file))
    index = csv_data.index.tolist()
    # read data and perform variance fusion
    csv_data = variance_fusion(csv_data)
    # break the given time-series into independent samples to test  based on annotation frequency
    csv_data = resample_data_labels(csv_data)
    # z-score normalize data
    csv_data = (csv_data - csv_data.mean()) / csv_data.std() 
    # get predictions
    valence_predictions = valence_model.predict(csv_data)[:, 1]
    arousal_predictions = arousal_model.predict(csv_data)[:, 1]
    predictions = np.dstack([valence_predictions, arousal_predictions]).squeeze()
    df = pd.DataFrame(predictions, columns=['valence', 'arousal'])
    df = df.set_index(pd.RangeIndex(min(index), max(index), 50))
    df.index.name = 'time'
    # align predictions based on the timestamps given in the test set
    df = df.loc[annotations['time']]
    # save test predictions for each input csv as a csv to be evaluated by the organizers
    df.to_csv(os.path.join(save_dir, file))    

  return
    
    
if __name__ == '__main__':
    # 5 because of 5-folds given in the challenge
    for fold in range(5):  
        # testing setup
        LOG_DIR = '../checkpoints/scenario-2/fold_{}'.format(fold)
        os.makedirs(LOG_DIR, exist_ok=True)
        test_fold(fid=fold, log_file=LOG_DIR)
        