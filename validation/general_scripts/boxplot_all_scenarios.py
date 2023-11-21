#%%
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define the scenarios
scenarios = ["scenario_1", "scenario_2", "scenario_3", "scenario_4"]

# Initialize an empty list to store the data
data = []
data_scenario_1 = []
data_aux = []

# Loop over the scenarios
for scenario in scenarios:
    with open(f"../../validation/{scenario}/regression_plot/performance_metrics.json", 'r') as f:
        performance_metrics = json.load(f)

    if scenario == "scenario_1":
        for file, metrics in performance_metrics.items():
            for model, model_metrics in metrics.items():
                for dimension, measures in model_metrics.items():
                    data_scenario_1.append([scenario, "fold_0", file, dimension, model, measures['rmse']])
                    if model != 'last value predictions':
                        data_aux.append([scenario, "fold_0", file, dimension, model, measures['rmse']])
                    data.append([scenario, "fold_0", file, dimension, model, measures['rmse']])
    else:
        for fold, fold_metrics in performance_metrics.items():
            for file, metrics in fold_metrics.items():
                for model, model_metrics in metrics.items():
                    for dimension, measures in model_metrics.items():
                        if model != 'last value predictions':
                            data_aux.append([scenario, fold, file, dimension, model, measures['rmse']])
                        data.append([scenario, fold, file, dimension, model, measures['rmse']])

# Transform the data into a pandas DataFrame suitable for the boxplot
df = pd.DataFrame(data_aux, columns=['Scenario', 'Fold', 'File', 'Dimension', 'Model', 'RMSE'])

# Filter DataFrame to only include 'predictions' and 'dummy regression' models
df_filtered = df[(df['Model'] == 'predictions') | (df['Model'] == 'dummy regressor')]

# Group by Scenario and Model, calculate mean RMSE
overall_rmse_per_scenario = df_filtered.groupby(['Scenario', 'Model'])['RMSE'].mean()

print(overall_rmse_per_scenario)

# Calculate and print the mean RMSE of each model across all scenarios
mean_rmse_across_scenarios = overall_rmse_per_scenario.groupby('Model').mean()
print(mean_rmse_across_scenarios)

df_scenario_1 = pd.DataFrame(data_scenario_1, columns=['Scenario', 'Fold', 'File', 'Dimension', 'Model', 'RMSE'])

# Separate dataframes for arousal and valence
df_arousal = df[df['Dimension'] == 'arousal']
df_valence = df[df['Dimension'] == 'valence']
df_arousal_scenario_1 = df_scenario_1[df_scenario_1['Dimension'] == 'arousal']
df_valence_scenario_1 = df_scenario_1[df_scenario_1['Dimension'] == 'valence']
def plot_rmse(df, dimension, title, filename):
    plt.figure(figsize=(12, 8))
    sns.set(style="whitegrid", font_scale=1.2)  # set style and increase font size
    ax = sns.boxplot(data=df, x='Scenario', y='RMSE', hue='Model', palette="Set3")  # Set palette to Set3
    sns.stripplot(data=df, x='Scenario', y='RMSE', hue='Model', jitter=True, size=4, dodge=True, color=".3", linewidth=0)  # Add stripplot

    handles, labels = ax.get_legend_handles_labels()
    n_models = len(df['Model'].unique())
    plt.legend(handles[:n_models], labels[:n_models], loc='center left', bbox_to_anchor=(1, 0.5))  # To avoid duplicate legend and locate it at middle left

    # Calculate mean RMSE for each group
    group_means = df.groupby(['Scenario', 'Model'])['RMSE'].mean().reset_index()

    # Add mean RMSE to the plot
    x_coords = np.arange(len(df['Scenario'].unique()))  # get the x coordinates for the scenarios
    for x in x_coords:
        for i, model in enumerate(df['Model'].unique()):
            # find the corresponding mean RMSE value
            mean_rmse = group_means.loc[(group_means['Scenario'] == df['Scenario'].unique()[x]) & (group_means['Model'] == model), 'RMSE']
            if not mean_rmse.empty:
                # Modify x position based on the number of models
                if n_models == 3:
                    plt.text(x - 0.3 + 0.3 * i, df['RMSE'].max(), f"{mean_rmse.values[0]:.2f}", ha='center')
                else:
                    plt.text(x - 0.2 + 0.4 * i, df['RMSE'].max(), f"{mean_rmse.values[0]:.2f}", ha='center')

    plt.title(title)
    plt.grid(True)
    plt.savefig(filename)
    plt.show()

# Create the boxplots for Arousal (all scenarios)
plot_rmse(df_arousal, 'arousal', 'Arousal prediction vs Dummy model (RMSE)', './rmse_comparison_arousal.png')

# Create the boxplots for Valence (all scenarios)
plot_rmse(df_valence, 'valence', 'Valece prediction vs Dummy model (RMSE)', './rmse_comparison_valence.png')

# Create the boxplots for Arousal (scenario_1)
plot_rmse(df_arousal_scenario_1, 'arousal', 'Arousal prediction vs Dummy model (RMSE) - Scenario 1', './rmse_comparison_arousal_scenario_1.png')

# Create the boxplots for Valence (scenario_1)
plot_rmse(df_valence_scenario_1, 'valence', 'Valence prediction vs Dummy model (RMSE) - Scenario 1', './rmse_comparison_valence_scenario_1.png')

# %%
df_aux = pd.DataFrame(data, columns=['Scenario', 'Fold', 'File', 'Dimension', 'Model', 'RMSE'])
# Create a DataFrame for scenario_1 data
df_scenario_1 = df_aux[df_aux['Scenario'] == 'scenario_1']

# Calculate mean RMSE for 'last value predictions' in scenario 1
last_value_predictions_rmse_scenario_1 = df_scenario_1[df_scenario_1['Model'] == 'last value predictions']['RMSE'].mean()

# Create a DataFrame for other scenarios
df_other_scenarios = df_aux[df_aux['Scenario'] != 'scenario_1']

# Calculate mean RMSE for 'dummy regressor' in other scenarios
dummy_regressor_rmse_other_scenarios = df_other_scenarios[df_other_scenarios['Model'] == 'dummy regressor'].groupby('Scenario')['RMSE'].mean()

# Print RMSE for 'last value predictions' in scenario 1
print('RMSE for last value predictions in scenario 1:', last_value_predictions_rmse_scenario_1)

# Print RMSE for 'dummy regressor' in other scenarios
print('\nRMSE for dummy regressor in other scenarios:')
print(dummy_regressor_rmse_other_scenarios)

# Print mean RMSE for 'dummy regressor' in other scenarios
print('\nMean RMSE for dummy regressor in other scenarios:', dummy_regressor_rmse_other_scenarios.mean())
# %%
# Calculate mean RMSE across all scenarios, considering 'last value predictions' for scenario 1
# and 'dummy regressor' for other scenarios

# Concatenate the RMSEs for the specific models in each scenario
all_rmse = pd.concat([pd.Series(last_value_predictions_rmse_scenario_1, index=['scenario_1']),
                      dummy_regressor_rmse_other_scenarios])

# Calculate mean
mean_rmse_across_all_scenarios = all_rmse.mean()

print('Mean RMSE across all scenarios:', mean_rmse_across_all_scenarios)

# %%
