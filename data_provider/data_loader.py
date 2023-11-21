import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
class Dataset_EPiC(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', scenario=1, vali_fold = 0, scenario_data_type = "train"):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 500
            self.label_len = 10
            self.pred_len = 10
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        # self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.scenario_data_type = scenario_data_type
        self.root_path = root_path
        self.data_path = data_path
        self.scenario = scenario
        self.vali_fold = f"fold_{vali_fold}"
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        self.lab_scaler = StandardScaler()
        if self.scenario==1:
            scenario_dir = f"{self.root_path}/scenario_{self.scenario}"
            storage = self.load_data_no_folds(scenario_dir,self.scenario_data_type)  
        else:
            scenario_dir = f"{self.root_path}/scenario_{self.scenario}"
            storage = self.load_data_with_folds(scenario_dir,self.scenario_data_type) 
        '''
        Split the feature and label into same timeframes
        '''
        if self.scenario==1:
            num_test = 200
            num_val = 200
            num_val_test = num_val+num_test

            overlap = self.label_len-1+ self.pred_len
            lab_pred_len = self.label_len-1+ self.pred_len

            
            train_feature = [np.array([feature.loc[feature.time.between(label.time[i-lab_pred_len],label.time[i])].iloc[:,1:].values\
                            for i in range(lab_pred_len,len(label)-num_val_test,overlap)] )\
                    for _, feature, label in tqdm(storage)]
            train_fea_timestamp = [np.array([feature.loc[feature.time.between(label.time[i-lab_pred_len],label.time[i])].iloc[:,:1].values\
                                        for i in range(lab_pred_len,len(label)-num_val_test,overlap)] )\
                                for _, feature, label in storage]

            valdation_feature = [np.array([feature.loc[feature.time.between(label.time[i-lab_pred_len],label.time[i])].iloc[:,1:].values\
                                    for i in range(len(label)-num_val_test,len(label)-num_test,overlap)] )\
                            for _, feature, label in storage]
            valdation_fea_timestamp = [np.array([feature.loc[feature.time.between(label.time[i-lab_pred_len],label.time[i])].iloc[:,:1].values\
                                        for i in range(len(label)-num_val_test,len(label)-num_test,overlap)] )\
                                for _, feature, label in storage]

            test_feature = [np.array([feature.loc[feature.time.between(label.time[i-lab_pred_len],label.time[i])].iloc[:,1:].values\
                                    for i in range(len(label)-num_test,len(label),self.pred_len)] )\
                            for _, feature, label in storage]
            test_fea_timestamp = [np.array([feature.loc[feature.time.between(label.time[i-lab_pred_len],label.time[i])].iloc[:,:1].values\
                                        for i in range(len(label)-num_test,len(label),self.pred_len)] )\
                                for _, feature, label in storage]

            train_lab = [np.array([label.iloc[i-lab_pred_len:i+1].iloc[:,1:].values\
                                    for i in range(lab_pred_len,len(label)-num_val_test,overlap)])\
                            for _, feature, label in storage]
            train_lab_timestamp = [np.array([label.iloc[i-lab_pred_len:i+1].iloc[:,:1].values\
                                            for i in range(lab_pred_len,len(label)-num_val_test,overlap)])\
                                for _, feature, label in storage]


            valdation_lab = [np.array([label.iloc[i-lab_pred_len:i+1].iloc[:,1:].values\
                                        for i in range(len(label)-num_val_test,len(label)-num_test,overlap)])\
                                for _, feature, label in storage]
            valdation_lab_timestamp = [np.array([label.iloc[i-lab_pred_len:i+1].iloc[:,:1].values\
                                                for i in range(len(label)-num_val_test,len(label)-num_test,overlap)])\
                                    for _, feature, label in storage]

    

            test_lab = [np.array([label.iloc[i-lab_pred_len:i+1].iloc[:,1:].values\
                                    for i in range(len(label)-num_test,len(label),self.pred_len)])\
                            for _, feature, label in storage]
            test_lab_timestamp = [np.array([label.iloc[i-lab_pred_len:i+1].iloc[:,:1].values\
                                            for i in range(len(label)-num_test,len(label),self.pred_len)])\
                                for _, feature, label in storage]
        else:
        
            num_validation = len(storage[self.vali_fold])//2
            fold_list = list(storage.keys())
            train_fold = [fl for fl in fold_list if fl != self.vali_fold]
            overlap = self.label_len-1+ self.pred_len

            lab_pred_len = self.label_len-1+ self.pred_len
            
            train_feature = [np.array([feature.loc[feature.time.between(label.time[i-lab_pred_len],label.time[i])].iloc[:,1:].values\
                            for i in range(lab_pred_len,len(label),overlap)])\
                   for fl in train_fold for _, feature, label in tqdm(storage[fl])]
            train_fea_timestamp = [np.array([feature.loc[feature.time.between(label.time[i-lab_pred_len],label.time[i])].iloc[:,:1].values\
                                        for i in range(lab_pred_len,len(label),overlap)] )\
                                for fl in train_fold for _, feature, label in storage[fl]]

            valdation_feature = [np.array([feature.loc[feature.time.between(label.time[i-lab_pred_len],label.time[i])].iloc[:,1:].values\
                                    for i in range(lab_pred_len,len(label),overlap)] )\
                            for _, feature, label in storage[self.vali_fold][:num_validation]]
            valdation_fea_timestamp = [np.array([feature.loc[feature.time.between(label.time[i-lab_pred_len],label.time[i])].iloc[:,:1].values\
                                        for i in range(lab_pred_len,len(label),overlap)] )\
                                for _, feature, label in storage[self.vali_fold][:num_validation]]

            test_feature = [np.array([feature.loc[feature.time.between(label.time[i-lab_pred_len],label.time[i])].iloc[:,1:].values\
                                    for i in range(lab_pred_len,len(label),self.pred_len)] )\
                            for _, feature, label in storage[self.vali_fold][num_validation:]]
            test_fea_timestamp = [np.array([feature.loc[feature.time.between(label.time[i-lab_pred_len],label.time[i])].iloc[:,:1].values\
                                        for i in range(lab_pred_len,len(label),self.pred_len)])\
                                for _, feature, label in storage[self.vali_fold][num_validation:]]


            train_lab = [np.array([label.iloc[i-lab_pred_len:i+1].iloc[:,1:].values\
                                    for i in range(lab_pred_len,len(label),overlap)])\
                            for fl in train_fold for _, feature, label in storage[fl]]
            train_lab_timestamp = [np.array([label.iloc[i-lab_pred_len:i+1].iloc[:,:1].values\
                                            for i in range(lab_pred_len,len(label),overlap)])\
                               for fl in train_fold  for _, feature, label in storage[fl]]


            valdation_lab = [np.array([label.iloc[i-lab_pred_len:i+1].iloc[:,1:].values\
                                        for i in range(lab_pred_len,len(label),overlap)])\
                                for _, feature, label in storage[self.vali_fold][:num_validation]]
            valdation_lab_timestamp = [np.array([label.iloc[i-lab_pred_len:i+1].iloc[:,:1].values\
                                                for i in range(lab_pred_len,len(label),overlap)])\
                                    for _, feature, label in storage[self.vali_fold][:num_validation]]


            test_lab = [np.array([label.iloc[i-lab_pred_len:i+1].iloc[:,1:].values\
                                    for i in range(lab_pred_len,len(label),self.pred_len)])\
                            for _, feature, label in storage[self.vali_fold][num_validation:]]
            test_lab_timestamp = [np.array([label.iloc[i-lab_pred_len:i+1].iloc[:,:1].values\
                                            for i in range(lab_pred_len,len(label),self.pred_len)])\
                                for _, feature, label in storage[self.vali_fold][num_validation:]]
            
        # print(train_feature[0].shape,valdation_feature[0].shape,test_feature[0].shape)
        train_feature=np.vstack(train_feature)
        valdation_feature=np.vstack(valdation_feature)
        test_feature=np.vstack(test_feature)
        # print(train_feature.shape,valdation_feature.shape,test_feature.shape)
        train_lab=np.vstack(train_lab)
        valdation_lab=np.vstack(valdation_lab)
        test_lab=np.vstack(test_lab)


        train_fea_timestamp=np.vstack(train_fea_timestamp)
        valdation_fea_timestamp=np.vstack(valdation_fea_timestamp)
        test_fea_timestamp=np.vstack(test_fea_timestamp)

        train_lab_timestamp=np.vstack(train_lab_timestamp)
        valdation_lab_timestamp=np.vstack(valdation_lab_timestamp)
        test_lab_timestamp=np.vstack(test_lab_timestamp)

        train_size = train_feature.shape[0]
        valdation_size = valdation_feature.shape[0]
        test_size = test_feature.shape[0]

        feature_all = np.vstack([train_feature,valdation_feature,test_feature])
        label_all = np.vstack([train_lab,valdation_lab,test_lab])

        fea_timestamp_all = np.vstack([train_fea_timestamp,valdation_fea_timestamp,test_fea_timestamp])
        lab_timestamp_all = np.vstack([train_lab_timestamp,valdation_lab_timestamp,test_lab_timestamp])


        border1s = [0, train_size, feature_all.shape[0]-test_size]
        border2s = [train_size, train_size + valdation_size, feature_all.shape[0]]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.scale:
            train_data = feature_all[border1s[0]:border2s[0]]
            # train_label = label_all[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.reshape(-1, train_data.shape[-1]))
            # self.lab_scaler.fit(train_label.reshape(-1, train_label.shape[-1]))
            data = self.scaler.transform(feature_all.reshape(-1, feature_all.shape[-1])).reshape(feature_all.shape)
            # label_scale = self.lab_scaler.transform(label_all.reshape(-1, label_all.shape[-1])).reshape(label_all.shape)
        else:
            data = feature_all
            label_scale = label_all

        self.data_x = data[border1:border2]
        self.data_y = label_all[border1:border2]
        self.data_stamp_x = fea_timestamp_all[border1:border2]
        self.data_stamp_y = lab_timestamp_all[border1:border2]

    def __getitem__(self, index):
        seq_x = self.data_x[index]
        seq_y = self.data_y[index]
        seq_x_mark = self.data_stamp_x[index]
        seq_y_mark = self.data_stamp_y[index]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x)

    def inverse_transform(self, data):
        return self.lab_scaler.inverse_transform(data)
    
    def load_data_no_folds(self,scenario_dir_path, dataset_type):
    # make dict to store data
        storage_list = list()
        # make paths for the specified dataset
        train_annotations_dir = Path(scenario_dir_path, dataset_type, "annotations")
        train_physiology_dir = Path(scenario_dir_path, dataset_type, "physiology")
        # sort contents of dirs, so that physiology and annotations are in the same order  
        train_physiology_files = sorted(Path(train_physiology_dir).iterdir())
        train_annotation_files = sorted(Path(train_annotations_dir).iterdir())
        # iterate over annotation and physiology files

        storage_list = [(annotations_file_path.name, pd.read_csv(physiology_file_path), pd.read_csv(annotations_file_path))\
                    for physiology_file_path, annotations_file_path in zip(train_physiology_files, train_annotation_files)]
        # for physiology_file_path, annotations_file_path in zip(train_physiology_files, train_annotation_files):
        #     # make sure that we load corresponding physiology and annotations
        #     assert physiology_file_path.name == annotations_file_path.name, "Order mismatch"
        #     # load data from files
        #     df_physiology = pd.read_csv(physiology_file_path)
        #     df_annotations = pd.read_csv(annotations_file_path)
        #     # continue # comment / delete this line if you want to store data in data_store list
        #     # store data
        #     storage_list.append((annotations_file_path.name, df_physiology, df_annotations))
        return storage_list
    
    def load_data_with_folds(self,scenario_dir_path, dataset_type):
    # make dict to store data
        storage_dict = dict()
        # iterate over the scenario directory
        for fold_dir in Path(scenario_dir_path).iterdir():
            # make paths for current fold
            train_annotations_dir = Path(fold_dir, f"{dataset_type}/annotations/")
            train_physiology_dir = Path(fold_dir, f"{dataset_type}/physiology/")
            # make key in a dict for current fold 
            storage_dict.setdefault(fold_dir.name, list())
            # sort contents of dirs, so that physiology and annotations are in the same order  
            train_physiology_files = sorted(Path(train_physiology_dir).iterdir())
            train_annotation_files = sorted(Path(train_annotations_dir).iterdir())
            # iterate over annotation and physiology files
            for physiology_file_path, annotations_file_path in zip(train_physiology_files, train_annotation_files):
                # make sure that we load corresponding physiology and annotations
                assert physiology_file_path.name == annotations_file_path.name, "Order mismatch"
                # load data from files
                df_physiology = pd.read_csv(physiology_file_path)
                df_annotations = pd.read_csv(annotations_file_path)
                # continue # comment / delete this line if you want to store data in data_store list
                # store data
                storage_dict[fold_dir.name].append((annotations_file_path.name, df_physiology, df_annotations))
        return storage_dict

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)