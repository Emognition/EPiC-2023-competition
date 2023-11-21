#%%
import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
from glob import glob

import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
from glob import glob

# your existing functions are here

# Define the number of folds for each scenario
scenario_folds = {"scenario_2": 5, "scenario_3": 4, "scenario_4": 2}

# Loop over scenarios
for scenario, num_folds in scenario_folds.items():
    # Create directory for this scenario
    scenario_dir = f'../{scenario}/regression_plot'
    os.makedirs(scenario_dir, exist_ok=True)

    # Initialize dictionary for performance metrics
    performance_metrics = {}

    # Initialize DummyRegressor
    dummy_regressor = DummyRegressor(strategy='mean')

    # Loop over folds
    for fold in range(num_folds):
        # Get all files in the directory
        files = os.listdir(f"../../data/test_set/{scenario}/fold_{fold}/test/annotations/")

        # Initialize dictionary for this fold
        performance_metrics[f"fold_{fold}"] = {}

        # Loop over files
        for file in files:
            # Ignore non-csv files
            if not file.endswith('.csv'):
                continue

            print(f'Processing file: {file}')

            # Load the data
            df_train = pd.read_csv(f"../../data/train_set/{scenario}/fold_{fold}/train/annotations/{file}")
            df_test = pd.read_csv(f"../../data/test_set/{scenario}/fold_{fold}/test/annotations/{file}")
            df_pred = pd.read_csv(f"../../results/{scenario}/fold_{fold}/test/annotations/{file}")

            # Create df_dummy_regressor with mean of train data as arousal and valence
            df_dummy_regressor = df_pred.copy()
            for emotion in ['arousal', 'valence']:
                dummy_regressor.fit(df_train['time'].values.reshape(-1, 1), df_train[emotion])
                df_dummy_regressor[emotion] = dummy_regressor.predict(df_test['time'].values.reshape(-1, 1))

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
                plot_true_vs_predicted(scenario_dir, file, f"fold_{fold}", df_test['time'], df_test[emotion], df_pred[emotion], emotion.capitalize())

    # Save performance metrics as JSON
    with open(f'{scenario_dir}/performance_metrics.json', 'w') as f:
        json.dump(performance_metrics, f, indent=4)


# %%

