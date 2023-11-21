import glob
import re
import os

import random

import numpy as np
import pandas as pd

import contextlib
import joblib
from joblib import Parallel, delayed

from scipy.signal import resample
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor

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

def create_folder_structure(root_physiology_folder, root_annotations_folder, save_output_folder, scenario, fold=None, test=False):
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

    output_folder = os.path.join(save_output_folder, scenario_str, fold_str, 'test','annotations')

    # Create directories if they don't exist
    for folder in [phys_folder_train, ann_folder_train, phys_folder_test, ann_folder_test, output_folder]:
        if folder is not None:
            os.makedirs(folder, exist_ok=True)
            
    if test:
        return phys_folder_train, ann_folder_train, phys_folder_test, ann_folder_test, output_folder
    else:
        return phys_folder_train, ann_folder_train, output_folder


def save_files(x, y, file_path, phys_folder, ann_folder):
    file_name = os.path.basename(file_path).replace('.csv', '')
    subject_num, video_num = map(int, file_name.split('_')[1::2])    
    file_base_name = f'sub_{subject_num}_vid_{video_num}'
    
    np.save(os.path.join(phys_folder, file_base_name), x)
    np.save(os.path.join(ann_folder, file_base_name), y)
    
def get_subs_vids(folder_path):
    subs = set()
    vids = set()

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            match = re.match(r"sub_(\d+)_vid_(\d+).csv", file_name)
            if match:
                sub, vid = match.groups()
                subs.add(int(sub))
                vids.add(int(vid))

    return sorted(list(subs)), sorted(list(vids))

def load_and_concatenate_train(base_path, vid, split = None):
    subjects, videos = get_subs_vids(base_path)
    # select_dict = {"subjects": subjects, "videos": videos}
    
    if split == None:
        data = []
        for sub in subjects:
            file_path = os.path.join(base_path, f"sub_{sub}_vid_{vid}.npy")
            data = np.load(file_path)
        return data
    
    else:
        train_data = []

        for train_test, subs in split.items():
            for sub in subs:
                file_path = os.path.join(base_path, f"sub_{sub}_vid_{vid}.npy")
                if os.path.exists(file_path):
                    data = np.load(file_path)
                    if train_test == "train":
                        train_data.append(data)

        train_data = np.concatenate(train_data) if train_data else None

        return train_data
    
def split_subjects_train_test(subjects, splits):
    """Split subjects into train and test sets.

    Args:
        subjects (_type_): _description_
        splits (_type_): _description_

    Returns:
        splits: list of dictionaries with keys 'train' and 'test' and values as lists of subject numbers.
    """
    def partition (list_in, n):
        random.shuffle(list_in)
        return [list_in[i::n] for i in range(n)]
    
    partitions  = partition(subjects, 3)
    
    splits = []

    for i in partitions:
        train = [x for x in subjects if x not in i]
        test = i
        
        splits.append({'train': train, 'test': test})
        
    return splits


    
    
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

def test_pipeline(X_train, X_test, y_train, y_test, model, numeric_column_indices=None,test = True):
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

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model_instance)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    if test:
        return y_pred
    
    else:
        # Calculate RMSE for each output separately
        importances = pipeline.named_steps['model'].estimators_[0].feature_importances_
        rmse_per_output = mean_squared_error(y_test, y_pred, squared=False, multioutput='raw_values')

        return y_pred, rmse_per_output, importances


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

    y_test_file = y_test_file.replace('npy', 'csv')
    
    if test:
        df = pd.read_csv(y_test_file)
        df['valence'] = y_pred[:, 0]
        df['arousal'] = y_pred[:, 1]  
    else:
        df = pd.DataFrame(y_test, columns=['valence', 'arousal'])
        df['valence_pred'] = y_pred[:, 1]
        df['arousal_pred'] = y_pred[:, 0] 
        df['time'] = pd.read_csv(y_test_file).tail(df.shape[0])['time'].values     
    

    
    df.to_csv(os.path.join(output_folder, os.path.basename(y_test_file)), index=False)
    return None

from scipy.signal import butter, filtfilt,cheby1, cheby2, ellip

def low_pass_filter(data, cutoff_frequency, sampling_rate, order, filter_type='butter'):
    nyquist_freq = 0.5 * sampling_rate
    normal_cutoff = cutoff_frequency / nyquist_freq

    if filter_type == 'butter':
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
    elif filter_type == 'cheby1':
        ripple = 1  # Adjust this value to change the passband ripple (in dB)
        b, a = cheby1(order, ripple, normal_cutoff, btype='low', analog=False)
    elif filter_type == 'cheby2':
        stopband_attenuation = 40  # Adjust this value to change the stopband attenuation (in dB)
        b, a = cheby2(order, stopband_attenuation, normal_cutoff, btype='low', analog=False)
    elif filter_type == 'ellip':
        ripple = 1  # Adjust this value to change the passband ripple (in dB)
        stopband_attenuation = 40  # Adjust this value to change the stopband attenuation (in dB)
        b, a = ellip(order, ripple, stopband_attenuation, normal_cutoff, btype='low', analog=False)
    else:
        raise ValueError("Invalid filter_type. Choose from 'butter', 'cheby1', 'cheby2', or 'ellip'.")

    num_columns = data.shape[1]
    filtered_data = np.empty_like(data)

    for col_idx in range(num_columns):
        filtered_data[:, col_idx] = filtfilt(b, a, data[:, col_idx])

    return filtered_data

def moving_average(data, window_size):
    num_columns = data.shape[1]
    num_rows = data.shape[0]
    filtered_data = np.empty_like(data)

    for col_idx in range(num_columns):
        cumsum = np.cumsum(data[:, col_idx])
        cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
        filtered_data[window_size - 1:, col_idx] = cumsum[window_size - 1:] / window_size

        # Handle the edge cases for the first window_size points
        for i in range(window_size - 1):
            filtered_data[i, col_idx] = cumsum[i] / (i + 1)

    return filtered_data

from scipy.signal import medfilt

def median_filter(data, window_size):
    num_columns = data.shape[1]
    filtered_data = np.empty_like(data)

    for col_idx in range(num_columns):
        padded_data = np.pad(data[:, col_idx], (window_size // 2, window_size // 2), mode='reflect')
        filtered_data[:, col_idx] = medfilt(padded_data, window_size)[window_size // 2: -(window_size // 2)]

    return filtered_data

from scipy.signal import savgol_filter
def savitzky_golay_multi_output(data, window_size, polynomial_order = 4):
    num_columns = data.shape[1]
    filtered_data = np.empty_like(data)

    for col_idx in range(num_columns):
        filtered_data[:, col_idx] = savgol_filter(
            data[:, col_idx],
            window_size,
            polynomial_order,
            mode='mirror'  # Reflect the input signal at the edges
        )

    return filtered_data

import numpy as np
from scipy.ndimage import gaussian_filter

def gaussian_filter_multi_output(data, sigma):
    num_columns = data.shape[1]
    filtered_data = np.empty_like(data)

    for col_idx in range(num_columns):
        filtered_data[:, col_idx] = gaussian_filter(data[:, col_idx], sigma, mode='reflect')

    return filtered_data
