import glob
import re
import os

import random

import numpy as np
import pandas as pd

import contextlib
import joblib
from joblib import Parallel, delayed
import pysiology

from scipy.signal import resample
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.decomposition import PCA

def zip_csv_files(folder_path_1, folder_path_2):
    """reads all csv files in the folder and returns a list of tuples with corresponding CSV file paths in both folders. Useful to loop over all files in two folders.

    Args:
        folder_path_1 (_type_): _description_
        folder_path_2 (_type_): _description_

    Returns:
        zipped_files: (tuple) list of tuples with corresponding CSV file paths in both folders
    """
    files_1 = glob.glob(folder_path_1 + '/*.csv')
    files_2 = glob.glob(folder_path_2 + '/*.csv')

    # Create a dictionary with keys as (subject_num, video_num) and values as the file path
    files_dict_1 = {(int(s), int(v)): f for f in files_1 for s, v in re.findall(r'sub_(\d+)_vid_(\d+)', f)}
    files_dict_2 = {(int(s), int(v)): f for f in files_2 for s, v in re.findall(r'sub_(\d+)_vid_(\d+)', f)}

    # Create a list of tuples with corresponding CSV file paths in both folders
    zipped_files = [(files_dict_1[key], files_dict_2[key]) for key in files_dict_1 if key in files_dict_2]

    return zipped_files

def zip_csv_train_test_files(folder_phys_train, folder_ann_train, folder_phys_test, folder_ann_test, format = '.csv'):
    """reads all csv or npy files in the folder and returns a list of tuples with corresponding CSV file paths in both folders. Useful to loop over all files in two folders.

    Args:
        folder_path_1 (_type_): _description_
        folder_path_2 (_type_): _description_

    Returns:
        zipped_files: (tuple) list of tuples with corresponding CSV file paths in both folders
    """
    if format == '.csv':
        files_phys_train = glob.glob(folder_phys_train + '/*.csv')
        files_ann_train = glob.glob(folder_ann_train + '/*.csv')
        files_phys_test = glob.glob(folder_phys_test + '/*.csv')
        files_ann_test = glob.glob(folder_ann_test + '/*.csv')

    elif format == '.npy':
        files_phys_train = glob.glob(folder_phys_train + '/*.npy')
        files_ann_train = glob.glob(folder_ann_train + '/*.npy')
        files_phys_test = glob.glob(folder_phys_test + '/*.npy')
        files_ann_test = glob.glob(folder_ann_test + '/*.npy')


    # Create a dictionary with keys as (subject_num, video_num) and values as the file path
    files_dict_phys_train = {(int(s), int(v)): f for f in files_phys_train for s, v in re.findall(r'sub_(\d+)_vid_(\d+)', f)}
    files_dict_ann_train = {(int(s), int(v)): f for f in files_ann_train for s, v in re.findall(r'sub_(\d+)_vid_(\d+)', f)}
    
    files_dict_phys_test = {(int(s), int(v)): f for f in files_phys_test for s, v in re.findall(r'sub_(\d+)_vid_(\d+)', f)}
    files_dict_ann_test = {(int(s), int(v)): f for f in files_ann_test for s, v in re.findall(r'sub_(\d+)_vid_(\d+)', f)}

    # Create a list of tuples with corresponding CSV file paths in both folders
    zipped_files_train = [(files_dict_phys_train[key], files_dict_ann_train[key]) for key in files_dict_phys_train if key in files_dict_ann_train]
    zipped_files_test = [(files_dict_phys_test[key], files_dict_ann_test[key]) for key in files_dict_phys_test if key in files_dict_ann_test]
    
    zipped_dict = {'train': zipped_files_train, 'test': zipped_files_test}
    

    return zipped_dict

def create_folder_structure(root_physiology_folder, root_annotations_folder, save_output_folder, scenario, fold=None, test=False, version=''):
    # Create scenario path
    scenario_str = f"scenario_{scenario}"
    
    # Create fold path if fold is not None
    fold_str = "" if fold is None else f"fold_{fold}"

    # Create paths
    if test:
        phys_folder_train = os.path.join(root_physiology_folder, scenario_str, fold_str, "train", "physiology")
        ann_folder_train = os.path.join(root_annotations_folder, scenario_str, fold_str, "train", "annotations")
        phys_folder_test = os.path.join(root_physiology_folder, scenario_str, fold_str, "test", "physiology")
        ann_folder_test = os.path.join(root_annotations_folder, scenario_str, fold_str, "test", "annotations")
    else:
        phys_folder_train = os.path.join(root_physiology_folder, scenario_str, fold_str, "physiology")
        ann_folder_train = os.path.join(root_annotations_folder, scenario_str, fold_str, "annotations")
        phys_folder_test = None
        ann_folder_test = None

    output_folder = os.path.join(save_output_folder, scenario_str, fold_str, 'test','annotations',version)

    # Create directories if they don't exist
    for folder in [phys_folder_train, ann_folder_train, phys_folder_test, ann_folder_test, output_folder]:
        if folder is not None:
            os.makedirs(folder, exist_ok=True)
            
    if test:
        return phys_folder_train, ann_folder_train, phys_folder_test, ann_folder_test, output_folder
    else:
        return phys_folder_train, ann_folder_train, output_folder


def save_files(x, y, file_path, phys_folder, ann_folder):
    subject_num, video_num = map(int, file_path.split(os.path.sep)[-1].replace('.csv', '').split('_')[1::2])
    
    file_base_name = f'sub_{subject_num}_vid_{video_num}'
    
    np.save(os.path.join(phys_folder, file_base_name), x)
    np.save(os.path.join(ann_folder, file_base_name), y)
    
        
def load_and_concatenate_files(base_path, train_test_split, vid ):
    train_data = []
    test_data = []

    for train_test, subs in train_test_split.items():
        if type(vid) == 'list':
            for v in vid:
                for sub in subs:
                    file_path = os.path.join(base_path, f"sub_{sub}_vid_{v}.npy")
                    if os.path.exists(file_path):
                        data = np.load(file_path)
                        if train_test == "train":
                            train_data.append(data)
                        else:
                            test_data.append(data)

        else:
            for sub in subs:
                file_path = os.path.join(base_path, f"sub_{sub}_vid_{vid}.npy")
                if os.path.exists(file_path):
                    data = np.load(file_path)
                    if train_test == "train":
                        train_data.append(data)
                    else:
                        test_data.append(data)
    

def preprocess(df_physiology, df_annotations, predictions_cols  = ['arousal', 'valence'], window = [-1000, 500], partition_window = 3, downsample_window = 10):
    # Función para generar features en base a la columna
    
    def identity(x):
        return x
    
    def partition_and_aggregate(arr, agg_func, partition_window):
        partition_size = arr.shape[1] // partition_window
        partitions = [arr[:, i * partition_size:(i + 1) * partition_size] for i in range(partition_window)]
        partitions_aggregated = [np.apply_along_axis(agg_func, axis=1, arr=partition) for partition in partitions]
        return np.concatenate(partitions_aggregated, axis=1)
    
    def generate_features(col_data, col_name, sr=500, partition_window=3):
        if col_name == 'ecg_cleaned':
            features = pysiology.electrocardiography.analyzeECG(col_data, samplerate = sr/downsample_window, preprocessing=False)
        elif col_name == 'gsr_cleaned':
            features = pysiology.electrodermalactivity.analyzeGSR(col_data, samplerate = sr/downsample_window, preprocessing=False)
        elif col_name in ['emg_zygo_cleaned', 'emg_coru_cleaned', 'emg_trap_cleaned']:
            features = pysiology.electromyography.analyzeEMG(col_data, samplerate = sr/downsample_window, preprocessing=False)
        
        # Aquí puedes definir tu función de agregación. En este ejemplo, estoy usando la media numpy,
        # pero puedes reemplazarla por cualquier función de agregación que desees utilizar.
        agg_func = np.mean()
        features = partition_and_aggregate(features, agg_func, partition_window)
        
        return features

    # Preprocesamiento inicial
    df_physiology = df_physiology[['time', 'ecg_cleaned', 'gsr_cleaned',  'emg_zygo_cleaned', 'emg_coru_cleaned', 'emg_trap_cleaned']]
    df_physiology['time'] = pd.to_timedelta(df_physiology['time'], unit='ms')
    df_physiology.set_index('time', inplace=True)
    df_annotations['time'] = pd.to_timedelta(df_annotations['time'], unit='ms')

    # Creación de ventanas deslizantes
    X_windows =  sliding_window_with_annotation(df_physiology, df_annotations, start=window[0], end=window[1], downsample = downsample_window)

    # Preparación de listas para almacenar features y sus nombres
    X_features = []
    feature_names = []

    # Para cada columna, generar features y añadirlas a la lista
    for i in range(X_windows.shape[2]):
        col_data = X_windows[:, :, i]
        col_name = df_physiology.columns[i]

        # Generar y almacenar features
        features = generate_features(col_data, col_name)
        X_features.append(features)
        
        # Generar y almacenar nombres de las features
        if isinstance(features, dict):
            for key in features.keys():
                feature_names.append(f'{col_name}_{key}')
        elif isinstance(features, dict) and all(isinstance(value, dict) for value in features.values()):
            for key_outer in features.keys():
                for key_inner in features[key_outer].keys():
                    feature_names.append(f'{col_name}_{key_outer}_{key_inner}')

    # Convertir listas a np.array
    X = np.array(X_features)
    y = df_annotations[predictions_cols].values

    return X, y, feature_names


def resample_data(x, downsample):
    len_x = len(x)
    num = len_x // downsample
    
    x_resample = resample(x, num=num, axis=0, domain='time')
    
    return x_resample
    

def process_annotation(arr, timestamps, annotation_time, start, end, window_size, downsample = 10):
    window_start_time = max(0, annotation_time + start)
    window_end_time = annotation_time + end

    mask = (timestamps >= window_start_time) & (timestamps <= window_end_time)
    
    window_data = arr[mask, :]
    
    resampled_window = resample_data(window_data, downsample)
    
    return resampled_window

def sliding_window_with_annotation(df, df_annotations, start=-1000, end=500, downsample = 10):
    df_annotations.set_index('time', inplace=True)
    window_size = abs(end - start) + 1

    # Convert index to integer (milliseconds)
    df.index = (df.index / pd.to_timedelta('1ms')).astype(int)
    df_annotations.index = (df_annotations.index / pd.to_timedelta('1ms')).astype(int)

    # Convert DataFrame to NumPy array
    arr = df.values
    timestamps = df.index.values

    # Initialize the time_adjusted_arr list and max_rows variable
    time_adjusted_arr = []
    max_rows = 0

    # Iterate through the annotations DataFrame
    for _, row in df_annotations.iterrows():
        annotation_time = row.name
        result = process_annotation(arr, timestamps, annotation_time, start, end, window_size, downsample)
        max_rows = max(max_rows, result.shape[0])
        time_adjusted_arr.append(result)

    # Pre-allocate the final_time_adjusted_arr with zeros
    final_time_adjusted_arr = np.zeros((len(df_annotations), max_rows, arr.shape[1]))

    # Fill the final_time_adjusted_arr with the data from time_adjusted_arr
    for i, result in enumerate(time_adjusted_arr):
        final_time_adjusted_arr[i, :result.shape[0], :] = result

    # print(f'final_time_adjusted_arr dimensions: {final_time_adjusted_arr.shape}')
    return final_time_adjusted_arr


def _fit_and_evaluate(train_index, test_index, X, y, pipeline):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Calculate RMSE for each output separately
    rmse_per_output = mean_squared_error(y_test, y_pred, squared=False, multioutput='raw_values')
    return rmse_per_output

from sklearn.multioutput import MultiOutputRegressor

def time_series_cross_validation_with_hyperparameters(X_train, X_test, y_train, y_test, model, numeric_column_indices=None,test = True, use_PCA=False, n_components=30):
    """
    Perform time series cross-validation with hyperparameters for a given model.

    Args:
        X (array-like): The feature matrix.
        y (array-like): The target vector.
        model (callable): The model class to be used.
        hyperparameters (dict): The hyperparameters for the model.
        n_splits (int, optional): The number of splits for time series cross-validation. Defaults to 5.
        n_jobs (int, optional): The number of parallel jobs to run. -1 means using all processors. Defaults to -1.
        numeric_column_indices (list, optional): The indices of numeric features. Defaults to None.
        categorical_column_indices (list, optional): The indices of categorical features. Defaults to None.

    Returns:
        float: The average root mean squared error (RMSE) across all splits.
    """

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # categorical_transformer = Pipeline(steps=[
    #     ('encoder', OneHotEncoder(handle_unknown='ignore'))
    # ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_column_indices),
        # ('cat', categorical_transformer, categorical_column_indices)
    ])

    # Check if y has multiple outputs
    multi_output = y_train.ndim > 1 and y_train.shape[1] > 1

    # Wrap the model in a MultiOutputRegressor if needed
    model_instance = model
    if multi_output:
        model_instance = MultiOutputRegressor(model_instance)

    # Add PCA to the pipeline if PCA is True
    steps = [
        ('preprocessor', preprocessor),
        ('model', model)
    ]
    if use_PCA:
        steps.insert(1, ('pca', PCA(n_components=n_components)))  # Adjust n_components as needed

    pipeline = Pipeline(steps)
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    # Calculate RMSE for each output separately
    if isinstance(model, Pipeline):
        model.fit(X_train, y_train)
        importances_df = pd.DataFrame(model.named_steps['model'].feature_importances_.reshape(1, -1))
    else:
        importances = pipeline.named_steps['model'].estimators_[0].feature_importances_
        importances_df = pd.DataFrame(importances.reshape(1, -1))

    if test:
        return y_pred, importances_df
    
    else:  
        rmse_per_output = mean_squared_error(y_test, y_pred, squared=False, multioutput='raw_values')

        return y_pred, importances_df,  rmse_per_output 


# Define the context manager
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()
        
def save_test_data(y_pred, output_folder, y_test_file, test = True, y_test = None):
    
    base_name = os.path.splitext(os.path.basename(y_test_file))[0]
    
    if test:
        df = pd.DataFrame(np.load(y_test_file), columns=['valence', 'arousal'])
        df['valence'] = y_pred[:, 1]
        df['arousal'] = y_pred[:, 0]
    else:
        df = pd.DataFrame(y_test, columns=['valence', 'arousal'])
        df['valence_pred'] = y_pred[:, 1]
        df['arousal_pred'] = y_pred[:, 0] 
        df['time'] = pd.read_csv(y_test_file.replace('npy', 'csv')).tail(df.shape[0])['time'].values     

    # Use the new base_name to save the file
    df.to_csv(os.path.join(output_folder, base_name + ".csv"), index=False)
    return None