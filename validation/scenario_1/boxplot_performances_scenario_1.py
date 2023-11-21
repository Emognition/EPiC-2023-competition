#%%

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json

version = 'PCA'

# Load the JSON file
with open(f'./regression_plot/{version}/performance_metrics.json', 'r') as f:
    performance_metrics = json.load(f)

# Transform the data into a pandas DataFrame suitable for the boxplot
data = []
for file, metrics in performance_metrics.items():
    for model, model_metrics in metrics.items():
        for dimension, measures in model_metrics.items():
            data.append([file, dimension, model, measures['rmse'], measures['ccc']])
df = pd.DataFrame(data, columns=['File', 'Dimension', 'Model', 'RMSE', 'ccc'])

# Separate dataframes for arousal and valence
df_arousal = df[df['Dimension'] == 'arousal']
df_valence = df[df['Dimension'] == 'valence']

# Calculate the mean RMSE for each model and dimension
mean_rmse_arousal = df_arousal.groupby('Model')['RMSE'].mean()
mean_rmse_valence = df_valence.groupby('Model')['RMSE'].mean()
print(mean_rmse_arousal)
print(mean_rmse_valence)


# Calculate the mean RMSE for each model and dimension
mean_ccc_arousal = df_arousal['ccc'].fillna(0).groupby(df_arousal['Model']).mean()
mean_ccc_valence = df_valence['ccc'].fillna(0).groupby(df_valence['Model']).mean()
print(mean_ccc_arousal)
print(mean_ccc_valence)


# Create the boxplots
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_arousal, x='Model', y='RMSE')
plt.title(f'RMSE for Different Models (Arousal) - CCC =  {mean_ccc_arousal["predictions"]:.2f}\n'
          f'Mean RMSE - Predictions: {mean_rmse_arousal["predictions"]:.2f}, '
          f'Last Value Predictions: {mean_rmse_arousal["last value predictions"]:.2f}, '
          f'Dummy Regressor: {mean_rmse_arousal["dummy regressor"]:.2f}')
plt.grid(True)
plt.savefig('./rmse_comparison_arousal.png')  # save the plot
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_valence, x='Model', y='RMSE')
plt.title(f'RMSE for Different Models (Arousal) - CCC =  {mean_ccc_valence["predictions"]:.2f}\n'
          f'Mean RMSE - Predictions: {mean_rmse_valence["predictions"]:.2f}, '
          f'Last Value Predictions: {mean_rmse_valence["last value predictions"]:.2f}, '
          f'Dummy Regressor: {mean_rmse_valence["dummy regressor"]:.2f}')
plt.grid(True)
plt.savefig(f'./{version}/rmse_comparison_valence.png')  # save the plot
plt.show()

# %%

