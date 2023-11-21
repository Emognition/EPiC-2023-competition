#%%
import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
from glob import glob

version = 'PCA'

def concordance_correlation_coefficient(y_true, y_pred):
    # Handle constant columns
    constant_columns_true = (y_true.std(axis=0) == 0)
    constant_columns_pred = (y_pred.std(axis=0) == 0)
    if constant_columns_true.any() or constant_columns_pred.any():
        print('Warning: y_true or y_pred contains constant columns.')
        return np.nan

    # Handle NaN values
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        print('Warning: y_true or y_pred contains NaN values.')
        return np.nan
    
    # Continue with CCC calculation
    cor = np.corrcoef(y_true, y_pred)[0, 1]
    
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)
    
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

    return numerator / denominator

def calculate_metrics(y_test, y_pred):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    ccc = concordance_correlation_coefficient(y_test, y_pred)
    return rmse, mae, r2, ccc

def plot_true_vs_predicted(file_name, time_train, y_train, time_test, y_test, y_pred, dummy_five, title):
    rmse, mae, r2, ccc= calculate_metrics(y_test, y_pred)
    
    # Concatenate train and test true values
    time = pd.concat([time_train, time_test])
    y_true = pd.concat([y_train, y_test])

    plt.figure(figsize=(10, 6))
    plt.plot(time, y_true, label='True Values', color='blue')
    plt.plot(time_test, y_pred, label='Predicted Values', color='red')
    plt.plot(time_test, dummy_five, label='last', color='black')
    plt.xlabel('Time')
    plt.ylabel(title)
    plt.title(f'{file_name} {title}: True vs Predicted')
    plt.grid(True)
    plt.ylim(0, 10) # Set y-axis limits
    plt.legend(loc='upper right')

    # Add performance metrics to the plot
    plt.text(x=0.05, y=0.95, s=f'RMSE = {rmse:.2f}\nMAE = {mae:.2f}\nR2 = {r2:.2f}\nCCC = {ccc:.2f}', 
             transform=plt.gca().transAxes, verticalalignment='top')
    
    # Dotted vertical line to indicate where test data starts
    plt.axvline(x=time_test.iloc[0], linestyle='dotted', color='black')
    
    plt.savefig(f'./regression_plot/{version}/{file_name}_{title}.png')  # save the plot
    plt.close()
    
    return rmse, mae, r2  # return metrics for each plot

os.makedirs(f'./regression_plot/{version}', exist_ok=True)

# Get all files in the directory
files = os.listdir(f"../../data/test_set/scenario_1/test/annotations")
print(files)

# Initialize dictionary for performance metrics
performance_metrics = {}

# Initialize DummyRegressor
dummy_regressor_arousal = DummyRegressor(strategy='constant', constant=5.0)	
dummy_regressor_valence = DummyRegressor(strategy='constant', constant=5.0)	

try:
    # your code here...
    for file in files:
        print(f"Processing {file}")
        # Ignore non-csv files
        if not file.endswith('.csv'):
            continue
    
        print(f'Processing file: {file}')

        # Load the data
        df_train = pd.read_csv(f"../../data/raw/scenario_1/train/annotations/{file}")
        df_test = pd.read_csv(f"../../data/test_set/scenario_1/test/annotations/{file}")
        df_pred = pd.read_csv(f"../../results/scenario_1/test/annotations/{version}/{file}")

        # Create df_dummy with constant arousal and valence
        df_dummy = df_pred.copy()
        df_dummy['arousal'] = df_train['arousal'].iloc[-1]
        df_dummy['valence'] = df_train['valence'].iloc[-1]

        dummy_regressor_arousal.fit(np.zeros(df_train['arousal'].shape), df_train['arousal'])
        dummy_regressor_valence.fit(np.zeros(df_train['valence'].shape), df_train['valence'])
        
        # Create df_dummy_regressor with mean of train data as arousal and valence
        df_dummy_regressor = df_pred.copy()
        df_dummy_regressor['arousal'] = dummy_regressor_arousal.predict(np.zeros(df_test.shape[0]))
        df_dummy_regressor['valence'] = dummy_regressor_valence.predict(np.zeros(df_test.shape[0]))

        # Assuming there is a "time" column in your csv
        time_train = df_train['time']
        # Generate new time series for test data
        start_time = time_train.iloc[-1] + 50
        end_time = start_time + len(df_test) * 50
        time_test = pd.Series(np.arange(start_time, end_time, 50))

        # Calculate and store performance metrics
        performance_metrics[file] = {}

        for model, df in zip(['predictions', 'last value predictions', 'dummy regressor'], [df_pred, df_dummy, df_dummy_regressor]):
            performance_metrics[file][model] = {}
            for emotion in ['arousal', 'valence']:
                rmse, mae, r2, ccc = calculate_metrics(df_test[emotion], df[emotion])
                performance_metrics[file][model][emotion] = {"rmse": rmse, "mae": mae, "r2": r2, "ccc": ccc}

        # Generate and save plots for only real predictions
        for emotion in ['arousal', 'valence']:
            try:
                rmse, mae, r2, ccc = calculate_metrics(df_test[emotion], df[emotion])
            except Exception as e:
                print(f"An error occurred while calculating metrics for {file}, {emotion}: {e}")

            plot_true_vs_predicted(file, time_train, df_train[emotion], time_test, df_test[emotion], df_pred[emotion], df_dummy[emotion], emotion.capitalize())

except Exception as e:
    print(f"An error occurred: {e}")
# Save performance metrics as JSON
with open(f'./regression_plot/{version}/performance_metrics.json', 'w') as f:
    json.dump(performance_metrics, f, indent=4)

print(performance_metrics)

# %%
