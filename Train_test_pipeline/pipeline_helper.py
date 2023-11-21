import glob,os
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from typing import List

def find_strings_with_substring(string_list: List[str], search_string: str) -> List[str]:
    """
    Find all strings in a list that contain a specified substring.

    Args:
        string_list (List[str]): A list of strings to search.
        search_string (str): The substring to search for.

    Returns:
        List[str]: A list of strings from the input list that contain the search string.
    """
    matching_strings = []
    for string in string_list:
        if search_string in string:
            matching_strings.append(string)
    return matching_strings

def get_file_paths(subjects, video_numbers, x_folder, y_folder, file_type = '1hz_zscored'):
    x_files = []
    y_files = []

    for subject in subjects:
        for video in video_numbers:
            # x_pattern = os.path.join(x_folder,
            #                          f"sub_{int(subject)}_vid_{int(video)}_processed_{file_type}.csv")
            x_pattern = os.path.join(x_folder,
                                     f"sub_{int(subject)}_vid_{int(video)}_*_{file_type}.csv")
            y_pattern = os.path.join(y_folder, f"sub_{int(subject)}_vid_{int(video)}.csv")
            #print(x_pattern)

            x_files.extend(glob.glob(x_pattern))
            y_files.extend(glob.glob(y_pattern))

    return x_files, y_files

def k_fold_cross_validation_time(subjects, video_numbers, x_folder, y_folder, file_type = '1hz'):
    train_x, train_y = get_file_paths(subjects, video_numbers, x_folder+'train/', y_folder+'train/', file_type)
    test_x, test_y = get_file_paths(subjects, video_numbers, x_folder+'test/', y_folder+'test/', file_type)

 
    fold_data = [((train_x, train_y), (test_x, test_y))]
    return fold_data

def k_fold_cross_validation(subjects, video_numbers, k, x_folder, y_folder, file_type = '1hz_zscored'):
    subject_indices = np.arange(len(subjects))
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    fold_data = []
    for train_index, test_index in kf.split(subject_indices):
        train_subjects = subjects[train_index]
        test_subjects = subjects[test_index]

        train_x, train_y = get_file_paths(train_subjects, video_numbers, x_folder, y_folder, file_type=file_type)
        test_x, test_y = get_file_paths(test_subjects, video_numbers, x_folder, y_folder,  file_type=file_type)

        fold_data.append(((train_x, train_y), (test_x, test_y)))

    return fold_data

def k_fold_cross_validation_video(subjects, video_numbers, k, x_folder, y_folder, file_type = '1hz_zscored'):
    video_indices = np.arange(len(video_numbers))
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    fold_data = []
    for train_index, test_index in kf.split(video_indices):
        train_videos = video_numbers[train_index]
        test_videos = video_numbers[test_index]

        train_x, train_y = get_file_paths(subjects, train_videos, x_folder, y_folder, file_type=file_type)
        test_x, test_y = get_file_paths(subjects, test_videos, x_folder, y_folder, file_type=file_type)

        fold_data.append(((train_x, train_y), (test_x, test_y)))

    return fold_data

import torch
from torch.utils.data import Dataset

class EmognitionDataset(Dataset):
    def __init__(self, x_files, y_files, x_columns=None, y_columns=None, downsampling_factor_x=1,downsampling_factor_y=1):
        self.x_data = []
        self.y_data = []  
        self.sub_num_data = []
        self.vid_num_data = []

        for file in x_files:
            x_df = pd.read_csv(file)
            x_array = x_df[x_columns].values[::downsampling_factor_x]
            self.x_data.append(x_array)

            sub_num_array = x_df['sub_num'].values[::downsampling_factor_x]
            vid_num_array = x_df['vid_num'].values[::downsampling_factor_x]
            self.sub_num_data.append(sub_num_array)
            self.vid_num_data.append(vid_num_array)
     

        for file in y_files:
            y_df = pd.read_csv(file, usecols=y_columns)
            y_array = y_df.values[::downsampling_factor_y]
            self.y_data.append(y_array)
            
            
        for i in range(len(self.x_data)):
            
            # make sure they are the same length between x_file and y_files
            min_len = min(len(self.x_data[i]), len(self.y_data[i]))
            self.x_data[i] = self.x_data[i][:min_len]
            self.y_data[i] = self.y_data[i][:min_len]
            self.sub_num_data[i] = self.sub_num_data[i][:min_len]
            self.vid_num_data[i] = self.vid_num_data[i][:min_len]
            
            # Combine x and y data, drop rows with NaN values
            combined_df = pd.concat([pd.DataFrame(self.x_data[i], columns=x_columns),
                         pd.DataFrame(self.y_data[i], columns=y_columns),
                         pd.DataFrame(self.sub_num_data[i], columns=['sub_num_dataset']),
                         pd.DataFrame(self.vid_num_data[i], columns=['vid_num_dataset'])], axis=1)

            # Drop rows with NaN values
            combined_df = combined_df.fillna(0)

            # Extract cleaned x, y, sub_num, and vid_num data
            self.x_data[i] = combined_df[x_columns].values
            self.y_data[i] = combined_df[y_columns].values
            self.sub_num_data[i] = combined_df['sub_num_dataset'].values
            self.vid_num_data[i] = combined_df['vid_num_dataset'].values
  
            
    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        x = self.x_data[index]
        y = self.y_data[index]
        sub_num = self.sub_num_data[index]
        vid_num = self.vid_num_data[index]

        return x, y, sub_num, vid_num


from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import joblib

def TrainModel(model, fold_data, x_columns, y_columns, save_file_name, 
                        downsampling_factor_y=20, 
                        downsampling_factor_x=1,verbose = True):
    '''
    which_features: x column names
    which_y: y column name
    save_file_name: descriptive name for the specific version of this model.
    e.g., LinearReg_arousal_
    
    '''
    test_res = []
    all_pred_y = []
    all_test_y = []
    all_sub_num = []
    all_vid_num = []
    for f, fold in enumerate(fold_data):
        train_x_files, train_y_files = fold[0]
        test_x_files, test_y_files = fold[1]

        train_dataset = EmognitionDataset(
            train_x_files, train_y_files, x_columns=x_columns, y_columns=y_columns,
            downsampling_factor_y=downsampling_factor_y,
            downsampling_factor_x=downsampling_factor_x
        )
        
        test_dataset = EmognitionDataset(
            test_x_files, test_y_files, x_columns=x_columns, y_columns=y_columns,
            downsampling_factor_y=downsampling_factor_y,
            downsampling_factor_x=downsampling_factor_x
        )

        train_x = np.vstack([train_dataset[i][0] for i in range(len(train_dataset))])
        train_y = np.vstack([train_dataset[i][1] for i in range(len(train_dataset))])

        test_x = np.vstack([test_dataset[i][0] for i in range(len(test_dataset))])
        test_y = np.vstack([test_dataset[i][1] for i in range(len(test_dataset))])
        #print([test_dataset[i][2] for i in range(len(test_dataset))])
        test_sub = np.hstack([test_dataset[i][2] for i in range(len(test_dataset))])
        test_vid = np.hstack([test_dataset[i][3] for i in range(len(test_dataset))])

        model.fit(train_x, train_y)
        
        pred_y_train = model.predict(train_x).reshape(-1,1)
        rmse_train = sqrt(mean_squared_error(train_y,pred_y_train))
        #test_res.append(rmse)
        r2_train = r2_score(train_y,pred_y_train)

        # Make predictions
        pred_y = model.predict(test_x).reshape(-1,1)
        rmse = sqrt(mean_squared_error(test_y,pred_y))
        test_res.append(rmse)
        r2 = r2_score(test_y, pred_y)

        if verbose:
            print(f'---------------------- fold {f+1} ---------------------------')
            print(f'train rmse = {rmse_train:.3f} test rmse = {rmse:.3f}')
            print(f'train r2 score = {r2_train:.3f} test r2 score = {r2:.3f}')

        # save: 
        all_test_y.append(test_y)
        all_pred_y.append(pred_y)
        all_sub_num.append(test_sub)
        all_vid_num.append(test_vid)

  #  return all_pred_y
    all_test_y_concat = np.vstack(all_test_y)
    all_pred_y_concat = np.vstack(all_pred_y)
    all_sub_num_concat = np.hstack(all_sub_num)
    all_vid_num_concat = np.hstack(all_vid_num)

    all_pred_y_df = pd.DataFrame({'sub': all_sub_num_concat, 'vid': all_vid_num_concat, 'pred_y': all_pred_y_concat[:,0],'test_y':all_test_y_concat[:,0]})
    all_pred_y_df.to_csv(save_file_name + '.csv', index=False)


    joblib.dump(model, save_file_name + '.joblib')
    return test_res, all_pred_y_df, model

def train_model_no_eval(model, 
                        train_x_files,
                        train_y_files,
                        x_columns, 
                        y_columns, 
                        save_file_name, 
                        downsampling_factor_y=20, 
                        downsampling_factor_x=1, 
                        verbose = True):
    '''
    which_features: x column names
    which_y: y column name
    save_file_name: descriptive name for the specific version of this model.
    e.g., LinearReg_arousal_
    
    '''
    train_x_files, train_y_files

    train_dataset = EmognitionDataset(
        train_x_files, 
        train_y_files,
        x_columns=x_columns,
        y_columns=y_columns,
        downsampling_factor_y=downsampling_factor_y,
        downsampling_factor_x=downsampling_factor_x
    )

    train_x = np.vstack([train_dataset[i][0] for i in range(len(train_dataset))])
    train_y = np.vstack([train_dataset[i][1] for i in range(len(train_dataset))])
    model.fit(train_x, train_y)
    joblib.dump(model, save_file_name + '.joblib')
    return model
        


