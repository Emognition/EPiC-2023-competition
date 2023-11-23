print("START")

import os
import time
import pandas as pd
import numpy as np
import re

# Data split & Tuning params
from sklearn.model_selection import PredefinedSplit, GroupKFold
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error

# XGBoost
from xgboost import XGBRegressor

# save model
import joblib

import hyperopt
from hyperopt import fmin, tpe, hp, anneal, Trials
from sklearn.model_selection import cross_val_score

############################################################################
# Define a function to get RMSE and save hyperopt results and the model:
def get_rmse_save_result (trials, best):
  # print condition
  if best is best1:
    condition = "numID"
  print("--------------------------")
  print("Results for "+condition+":")

  # Train the model on the full training set with the best hyperparameters
  best_model = XGBRegressor(random_state=random_state,
                            n_estimators=int(best['n_estimators']), eta=best['eta'],
                            max_depth=int(best['max_depth']), gamma=best['gamma'],
                            subsample=best['subsample'],
                            colsample_bytree=best['colsample_bytree'],
                            eval_metric="rmse", tree_method='gpu_hist', gpu_id=0)
  best_model.fit(X_total, y_total)

  # save the best model as a file
  joblib.dump(best_model, 'Model_s4_XGB_DF.joblib')
  
  # Save Hyperopt Iterations -----------------------------
  # extract
  rmse_df = pd.DataFrame(trials.results)
  rmse_df.columns = ['RMSE', 'status']
  # save
  result = pd.concat([pd.DataFrame(trials.vals), rmse_df], axis=1)
  result.to_csv('Hyper_s4_XGB_DF.csv', index=False)
  # Get summary statistics for the 'RMSE' column
  print("--- RMSE on TRAIN set ---")
  RMSE = result['RMSE'].describe()
  print(RMSE)
  # Print Mean and SD of RMSE
  print("Train RMSE Best:", RMSE['min'])
  print("Train RMSE Mean:", RMSE['mean'])
  print("Train RMSE SD:", RMSE['std'])

############################################################################

# Read files ------------------------------------------------------------

# Set directory of the processed data: change it to your own directory
path = "./"
os.chdir(path)

file_list = ["train_scenario_4_fold_0_dynfeatures.csv",
             "train_scenario_4_fold_1_dynfeatures.csv"]
total = pd.DataFrame()



for file_name in file_list:
    data = pd.read_csv(file_name)
    
    # print nrows
    print(data.shape[0])
    
    # Concatenate the data frames vertically
    total = pd.concat([total, data], axis=0)
    
# Handling Missing data -------------------------------------------------
total = total.groupby(['unique_number']).ffill().bfill()
# Drop features------------------------------------------
total.drop(['SCENARIO'], axis=1, inplace=True)

# Split Target vs. Predictors -------------------------------------------

# Select target & predictors
target_col = ['valence', 'arousal']
y_total = total[target_col]
X_total = total.drop(target_col, axis=1)


# # Custum CV ----------------------------------------------------------
test_fold = np.array([0] * 231787 + [1] * 218197)
test_fold
# The indices which have the value -1 will be kept in train.
# The indices which have zero or positive values, will be kept in test

# length of each fold:
# 231787
# 218197

ps = PredefinedSplit(test_fold)
# Check how many splits will be done, based on test_fold
ps.get_n_splits()

# OUTPUT: 1
for train_index, test_index in ps.split():
    print("TRAIN:", train_index, "TEST:", test_index)

# TUNE --------------------------------------------------------------------

# Set directory where the best model will be saved. Change it to your own directory
save_path = "./" 
os.chdir(save_path)


###### Hyperopt #######----------------------------------------------------
max_eval = 100 # number of iterations in hyperopt

# Step 1: Initialize space for exploring hyperparameters:
space={'max_depth': hp.quniform("max_depth", 3, 30, 1),
      'gamma': hp.uniform ('gamma', 0, 9),
      'subsample': hp.uniform('subsample', 0.5, 1),
      'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 1),
      'n_estimators': hp.quniform('n_estimators', 50, 300, 1),
      'learning_rate': hp.uniform ('learning_rate', 0.001, 0.5),
      'eta': hp.quniform('eta', 0.05, 0.5, 0.025)
      }
      
# # Toy space for testing code: use the space above for the real tuning
# space={
#       'max_depth': hp.quniform("max_depth", 3, 12, 1),
#       'gamma': hp.uniform ('gamma', 0, 0),
#       'subsample': hp.uniform('subsample', 0.99, 1),
#       'colsample_bytree': hp.uniform('colsample_bytree', 0.99, 1),
#       'n_estimators': hp.quniform('n_estimators', 3, 5, 1),
#       'eta': hp.quniform('eta', 0.2, 0.3, 0.025)
#     }

# Step 2: Define objective function:
def hyperparam_tuning1(space):
  model = XGBRegressor(n_estimators=int(space['n_estimators']),
                     eta=space['eta'],
                     max_depth=int(space['max_depth']),
                     gamma=space['gamma'],
                     subsample=space['subsample'],
                     colsample_bytree=space['colsample_bytree'],
                     eval_metric="rmse", seed=12435,tree_method='gpu_hist', gpu_id=0)
  # cross validation
  score = -cross_val_score(estimator=model, 
                                 X = X_total, y = y_total,
                                 cv=ps,
                                 scoring='neg_root_mean_squared_error',
                                 n_jobs = -1).mean()
  return score

# Step 3: Run Hyperopt function:
random_state=42
start = time.time()
trials1 = Trials()
best1 = fmin(fn=hyperparam_tuning1,
            space=space,
            algo=tpe.suggest,
            max_evals=max_eval,
            trials=trials1,
            rstate=np.random.default_rng(random_state))
print('It takes %s minutes' % ((time.time() - start)/60)) 
print ("Best params:", best1)

# Step 4: Get the results and save it:
get_rmse_save_result(trials1, best1)
