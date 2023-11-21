#%%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json

# Define the number of folds for each scenario
scenario_folds = {"scenario_2": 5, "scenario_3": 4, "scenario_4": 2}

# Initialize an empty list to hold the data
data = []

# Loop over scenarios
for scenario, num_folds in scenario_folds.items():
    # Load the JSON file
    with open(f'../{scenario}/regression_plot/performance_metrics.json', 'r') as f:
        performance_metrics = json.load(f)

    # Transform the data into a pandas DataFrame suitable for the boxplot
    for fold, metrics in performance_metrics.items():
        for file, file_metrics in metrics.items():
            for model, model_metrics in file_metrics.items():
                for dimension, measures in model_metrics.items():
                    data.append([scenario, fold, file, dimension, model, measures['rmse']])

# Create DataFrame from the collected data
df = pd.DataFrame(data, columns=['Scenario', 'Fold', 'File', 'Dimension', 'Model', 'RMSE'])

# Separate dataframes for arousal and valence
df_arousal = df[df['Dimension'] == 'arousal']
df_valence = df[df['Dimension'] == 'valence']

# Create the boxplots
for scenario in scenario_folds.keys():
    for dimension, df_dimension in [('Arousal', df_arousal), ('Valence', df_valence)]:
        df_scenario = df_dimension[df_dimension['Scenario'] == scenario]

        # Calculate the mean RMSE for each model
        mean_rmse = df_scenario.groupby('Model')['RMSE'].mean()

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_scenario, x='Model', y='RMSE')
        plt.title(f'RMSE for Different Models ({dimension}) - {scenario.capitalize()}\n'
                  f'Mean RMSE - Predictions: {mean_rmse["predictions"]:.2f}, '
                  f'Dummy Regressor: {mean_rmse["dummy regressor"]:.2f}')
        plt.grid(True)
        plt.savefig(f'../{scenario}/regression_plot/rmse_comparison_{dimension.lower()}.png')  # save the plot
        plt.show()
#%%