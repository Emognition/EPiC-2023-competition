#%%

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json

import matplotlib.pyplot as plt
import pandas as pd
import json
import seaborn as sns

# Load the JSON file
with open('./regression_plot/performance_metrics.json', 'r') as f:
    performance_metrics = json.load(f)

# Transform the data into a pandas DataFrame suitable for the boxplot
data = []
for fold, fold_metrics in performance_metrics.items():
    for file, metrics in fold_metrics.items():
        for model, model_metrics in metrics.items():
            for dimension, measures in model_metrics.items():
                data.append([fold, file, dimension, model, measures['rmse']])
df = pd.DataFrame(data, columns=['Fold', 'File', 'Dimension', 'Model', 'RMSE'])

# Separate dataframes for arousal and valence
df_arousal = df[df['Dimension'] == 'arousal']
df_valence = df[df['Dimension'] == 'valence']

# Calculate the mean RMSE for each model and dimension
mean_rmse_arousal = df_arousal.groupby('Model')['RMSE'].mean()
mean_rmse_valence = df_valence.groupby('Model')['RMSE'].mean()

# Create the boxplots
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_arousal, x='Model', y='RMSE')
plt.title(f'RMSE for Different Models (Arousal)\n'
          f'Mean RMSE - Predictions: {mean_rmse_arousal["predictions"]:.2f}, '
          f'Dummy Regressor: {mean_rmse_arousal["dummy regressor"]:.2f}')
plt.grid(True)
plt.savefig('./rmse_comparison_arousal.png')  # save the plot
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_valence, x='Model', y='RMSE')
plt.title(f'RMSE for Different Models (Valence)\n'
          f'Mean RMSE - Predictions: {mean_rmse_valence["predictions"]:.2f}, '
          f'Dummy Regressor: {mean_rmse_valence["dummy regressor"]:.2f}')
plt.grid(True)
plt.savefig('./rmse_comparison_valence.png')  # save the plot
plt.show()

# %%

