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
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from helpers_scenario4 import *

#config
scenario = 4

# create directory if it doesn't exist
results_dir = '../../results/scenario_4'
os.makedirs(results_dir, exist_ok=True)

folds = [0,1]
for fold in folds:
    root_physiology_folder = "../../data/preprocessed/"
    root_annotations_folder = "../../data/raw/"
    save_output_folder = "../../results/"


    phys_folder_train, ann_folder_train, phys_folder_test, ann_folder_test, output_folder, = create_folder_structure(
        root_physiology_folder, root_annotations_folder, save_output_folder, scenario, fold, test=True)


    zipped_dict = zip_csv_train_test_files(phys_folder_train, ann_folder_train, phys_folder_test, ann_folder_test, format = '.csv')
    # print(len(zipped_dict['train']))

    subjects, videos_train = get_subs_vids(phys_folder_train)
    subjects, videos_test = get_subs_vids(phys_folder_test)

    def process_files(annotation_file, physiology_file,):
        df_annotations = pd.read_csv(annotation_file)
        df_physiology = pd.read_csv(physiology_file)
        
        # print(physiology_file)
        X, y = preprocess(df_physiology, df_annotations,  predictions_cols=['arousal','valence'], aggregate=['mean','min'], window=[-5000, 5000], partition_window = 3)
        # print(X.shape, y.shape)
        
        save_files(X, y, annotation_file, os.path.dirname(physiology_file), os.path.dirname(annotation_file))
        
        return None


    # # # Process the files using the context manager
    for key in zipped_dict.keys():
        with parallel_backend('loky', n_jobs= multiprocessing.cpu_count()//2):
            with tqdm_joblib(tqdm(total=len(zipped_dict[key]), desc=f"{key} files", leave=False)) as progress_bar:
                results = Parallel()(
                    (delayed(process_files)(ann_file, phys_file) for phys_file, ann_file in zipped_dict[key])
                )


    # Define models and hyperparameters
    random_forest = RandomForestRegressor(
        n_estimators=100, 
        max_depth=None, 
        min_samples_split=5, 
        min_samples_leaf=1, 
        random_state = 42,
        n_jobs = -1
        # max_features='auto'
    )

    rf_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', MultiOutputRegressor(random_forest))
    ])
    
    num_cpu_cores = multiprocessing.cpu_count()

    def evaluate_features(sub):
        X_train  = load_and_concatenate_train(phys_folder_train, sub =sub,)
        y_train = load_and_concatenate_train(ann_folder_train, sub =sub)
        
        rf_pipeline.fit(X_train, y_train)
        importances = rf_pipeline.named_steps['rf'].estimators_[0].feature_importances_
        
        return importances

    all_importances = []
    with parallel_backend('loky', n_jobs=  num_cpu_cores - 5):
        with tqdm_joblib(tqdm(total=len(subjects), desc="Files", leave=False)) as progress_bar:
            results = Parallel()(
                (delayed(evaluate_features)(i) for i in subjects))
            
        # Combine results for all subjects
        for i in range(len(subjects)):
            all_importances.append(results[i])

    df_importances = pd.DataFrame(all_importances)

    # Take mean of importances (for every column)
    mean_importances = df_importances.mean(axis=0)

    # Select the indices of the best columns (highest mean)
    best_indices = mean_importances.nlargest(36).index

    def test_function(sub, rf_pipeline):
        X_train  = load_and_concatenate_train(phys_folder_train, sub =sub,)
        y_train = load_and_concatenate_train(ann_folder_train, sub =sub,)
        
        rf_pipeline.fit(X_train, y_train)
        

        for vid in videos_test:
            X_test = np.load(os.path.join(phys_folder_test, f"sub_{sub}_vid_{vid}.npy"))
            y_test = np.load(os.path.join(ann_folder_test, f"sub_{sub}_vid_{vid}.npy"))

            y_pred = rf_pipeline.predict(X_test)
            
            y_pred_filtered = gaussian_filter_multi_output(y_pred, 60)
            y_pred_filtered = low_pass_filter(y_pred_filtered, 1, 20,6)

            
            path_csv_test =  os.path.join(ann_folder_test, f"sub_{sub}_vid_{vid}.csv")

            save_test_data(y_pred_filtered, output_folder, path_csv_test, test = True)
            
        return None
    
    num_cpu_cores = multiprocessing.cpu_count()
    all_results = []
    all_importances = []
    with parallel_backend('loky', n_jobs=  num_cpu_cores - 5):
        with tqdm_joblib(tqdm(total=len(subjects), desc="Files", leave=False)) as progress_bar:
            results = Parallel()(
                (delayed(test_function)(i, rf_pipeline) for i in subjects))

# %%
