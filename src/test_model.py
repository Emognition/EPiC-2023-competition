import pandas as pd
from pathlib import Path
from scipy.ndimage import uniform_filter1d
from autogluon.tabular import TabularDataset, TabularPredictor
import warnings

warnings.filterwarnings("ignore")

def avg_filter(df, size=10):
    df.valence = uniform_filter1d(df.valence, size=size)
    df.arousal = uniform_filter1d(df.arousal, size=size)
    return df

def test(X, arousal_model_path, valence_model_path, save_path, late_fusion=False, average=10):
    """
        if late_fusion is True, model_path should be a list.
    """
    test_data = TabularDataset(X)

    # average the results from all models
    if late_fusion == False:
        arousal = []
        valence = []
        for one_model_path in arousal_model_path:
            arousal.append(TabularPredictor.load(str(one_model_path)).predict(test_data))
        for one_model_path in valence_model_path:
            valence.append(TabularPredictor.load(str(one_model_path)).predict(test_data))
        arousal = pd.concat(arousal, axis=1).mean(axis=1)
        valence = pd.concat(valence, axis=1).mean(axis=1)

    else:
        arousal = TabularPredictor.load(str(arousal_model_path)).predict(test_data)
        valence = TabularPredictor.load(str(valence_model_path)).predict(test_data)

    predictions = pd.DataFrame({'valence': valence, 'arousal': arousal})

    # smooth the prediction
    if average > 0:
        avg_filter(predictions, average)

    predictions.to_csv(save_path)
