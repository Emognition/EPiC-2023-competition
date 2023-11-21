import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor


def train(X, y, target, model_path, num_cpus, num_gpus):
    if target == "arousal":
        drop = "valence"
    else:
        drop = "arousal"

    train_data = pd.concat([X, y.drop([drop], axis=1)], axis=1)
    train_data = TabularDataset(train_data)
    predictor = TabularPredictor(
        label=target, problem_type="regression", path=str(model_path), verbosity=0
    ).fit(train_data, ag_args_fit={"num_cpus": num_cpus, "num_gpus": num_gpus})
