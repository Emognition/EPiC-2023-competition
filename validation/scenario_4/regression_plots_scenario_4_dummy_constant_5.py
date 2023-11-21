#%%
import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
from glob import glob

def calculate_metrics(y_test, y_pred):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return rmse, mae, r2

def plot_true_vs_predicted(file_name, fold, time_test, y_test, y_pred, title):
    rmse, mae, r2 = calculate_metrics(y_test, y_pred)
    
    plt.figure(figsize=(10, 6))
    plt.plot(time_test, y_test, label='True Values', color='blue')
    plt.plot(time_test, y_pred, label='Predicted Values', color='red')
    plt.xlabel('Time')
    plt.ylabel(title)
    plt.title(f'{file_name} {title}: True vs Predicted')
    plt.grid(True)
    plt.ylim(0, 10) # Set y-axis limits
    plt.legend(loc='upper right')

    # Add performance metrics to the plot
    plt.text(x=0.05, y=0.95, s=f'RMSE = {rmse:.2f}\nMAE = {mae:.2f}\nR2 = {r2:.2f}', 
             transform=plt.gca().transAxes, verticalalignment='top')
    
    plt.savefig(f'./regression_plot/{file_name}_{fold}_{title}.png')  # save the plot
    plt.close()
    
    return rmse, mae, r2  # return metrics for each plot

os.makedirs('./regression_plot', exist_ok=True)

# Initialize dictionary for performance metrics
performance_metrics = {}

# Initialize DummyRegressor
dummy_regressor_arousal = DummyRegressor(strategy='constant', constant=5.0)	
dummy_regressor_valence = DummyRegressor(strategy='constant', constant=5.0)	

# Loop over scenarios
for fold in range(2):
    # Get all files in the directory
    files = os.listdir(f"../../data/test_set/scenario_4/fold_{fold}/test/annotations/")

    # Initialize dictionary for this fold
    performance_metrics[f"fold_{fold}"] = {}

    # Load all the training data for this fold and calculate the mean of arousal and valence
    train_files = glob(f"../../data/raw/scenario_4/fold_{fold}/train/annotations/*.csv")
    df_train = pd.concat([pd.read_csv(f) for f in train_files])
    dummy_regressor_arousal.fit(np.zeros(df_train['arousal'].shape), df_train['arousal'])
    dummy_regressor_valence.fit(np.zeros(df_train['valence'].shape), df_train['valence'])

    # Loop over files
    for file in files:
        # Ignore non-csv files
        if not file.endswith('.csv'):
            continue

        print(f'Processing file: {file}')

        # Load the data
        df_test = pd.read_csv(f"../../data/test_set/scenario_4/fold_{fold}/test/annotations/{file}")
        df_pred = pd.read_csv(f"../../results/scenario_4/fold_{fold}/test/annotations/{file}")

        # Create df_dummy_regressor with mean of train data as arousal and valence
        df_dummy_regressor = df_pred.copy()
        df_dummy_regressor['arousal'] = dummy_regressor_arousal.predict(np.zeros(df_test.shape[0]))	
        df_dummy_regressor['valence'] = dummy_regressor_valence.predict(np.zeros(df_test.shape[0]))
        
        
        # Calculate and store performance metrics
        performance_metrics[f"fold_{fold}"][file] = {}

        for model, df in zip(['predictions', 'dummy regressor'], [df_pred, df_dummy_regressor]):
            performance_metrics[f"fold_{fold}"][file][model] = {}
            for emotion in ['arousal', 'valence']:
                rmse, mae, r2 = calculate_metrics(df_test[emotion], df[emotion])
                performance_metrics[f"fold_{fold}"][file][model][emotion] = {"rmse": rmse, "mae": mae, "r2": r2}

        # Generate and save plots for only real predictions
        for emotion in ['arousal', 'valence']:
            rmse, mae, r2 = calculate_metrics(df_test[emotion], df_pred[emotion])
            plot_true_vs_predicted(file, f"fold_{fold}", df_test['time'], df_test[emotion], df_pred[emotion], emotion.capitalize())

# Save performance metrics as JSON
with open('./regression_plot/performance_metrics.json', 'w') as f:
    json.dump(performance_metrics, f, indent=4)


# %%

