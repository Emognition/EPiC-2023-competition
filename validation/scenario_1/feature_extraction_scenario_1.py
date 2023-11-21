#%%
import os
import numpy as np
import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import joblib
from joblib import Parallel, delayed, parallel_backend
import multiprocessing
from multiprocessing import Value
from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def preprocess(df_physiology, df_annotations, predictions_cols  = 'arousal', aggregate=None, window = [-1000, 500], partition_window = 1, downsample_window = 10):
    """
    Preprocesses the input data for further processing and modeling.
    
    Parameters:
    ----------
    df_physiology : pd.DataFrame
        The input physiological data with columns: time, and physiological signals.
    df_annotations : pd.DataFrame
        The annotations DataFrame with columns: time and arousal.
    aggregate : list of str, optional
        The list of aggregation functions to apply on the input data.
        Available options are: 'mean', 'std', 'enlarged', or any combination of these.
        If None, it will return the 3D matrix as is.
    window_duration : list, optional
        The duration of the sliding window in milliseconds (default is -1000, 500).
    
    Returns:
    -------
    X : np.ndarray
        The preprocessed input data (features) as a 2D array.
    y : np.ndarray
        The arousal values corresponding to each window.
    numeric_column_indices : list of int
        The indices of numeric columns in the input data.
    categorical_column_indices : list of int
        The indices of categorical columns in the input data.
    """
    df_physiology['time'] = pd.to_timedelta(df_physiology['time'], unit='ms')
    df_physiology.set_index('time', inplace=True)
    df_annotations['time'] = pd.to_timedelta(df_annotations['time'], unit='ms')
    

    X_windows =  sliding_window_with_annotation(df_physiology, df_annotations, start=window[0], end=window[1], downsample = downsample_window)
    # print(f'X_windows dimensions: {X_windows.shape}')

    aggregate_local = aggregate.copy() if aggregate is not None else None

    X = np.array([np.array(X_windows[:, :, i].tolist()) for i in range(X_windows.shape[2])]).T
    
    # print('X shape: ', X.shape)
    
    
    def partition_and_aggregate(arr, agg_func, partition_window):
        partition_size = arr.shape[1] // partition_window
        partitions = [arr[:, i * partition_size:(i + 1) * partition_size] for i in range(partition_window)]
        partitions_aggregated = [np.apply_along_axis(agg_func, axis=1, arr=partition) for partition in partitions]
        return np.concatenate(partitions_aggregated, axis=1)

    X_aggregated = []

    if aggregate_local is not None:
        if "enlarged" in aggregate_local:
            X_enlarged = np.array([np.array(X_windows[i].flatten().tolist()) for i in range(X_windows.shape[0])])
            X_aggregated.append(X_enlarged)
            aggregate_local.remove("enlarged")

        agg_funcs = {
            "mean": np.mean,
            "std": np.std,
            "max": np.max,
            "min": np.min,
            # Add more aggregation functions here
        }

        for agg in aggregate_local:
            X_agg = partition_and_aggregate(X_windows, agg_funcs[agg], partition_window)
            X_aggregated.append(X_agg)

        if X_aggregated:
            X = np.concatenate(X_aggregated, axis=1)
    
    # print('X shape: ', X.shape)


    y = df_annotations[predictions_cols].values

    numeric_column_indices = [i for i, col_dtype in enumerate(df_physiology.dtypes) if np.issubdtype(col_dtype, np.number)]
    # categorical_column_indices = [i for i, col_dtype in enumerate(df_physiology.dtypes) if not np.issubdtype(col_dtype, np.number)]

    return X, y, #numeric_column_indices #,categorical_column_indices

scenario = 1

version = 'PCA'
components = 30
pca = True

fold = None

root_physiology_folder = "../../data/preprocessed/"
root_annotations_folder = "../../data/raw/"
save_output_folder = "../../results/"

phys_folder_train, ann_folder_train, phys_folder_test, ann_folder_test, output_folder, = create_folder_structure(
    root_physiology_folder, root_annotations_folder, save_output_folder, scenario, fold, test=True, version='PCA')

zipped_dict = zip_csv_train_test_files(phys_folder_train, ann_folder_train, phys_folder_test, ann_folder_test, format='.csv')

def process_files(annotation_file, physiology_file,):
    df_annotations = pd.read_csv(annotation_file)
    df_physiology = pd.read_csv(physiology_file)

    X, y = preprocess(df_physiology, df_annotations,  predictions_cols=['arousal','valence'], aggregate=['mean', 'std', 'max', 'min'], window=[-5000, 5000], partition_window=3)
    save_files(X, y, annotation_file, os.path.dirname(physiology_file), os.path.dirname(annotation_file))