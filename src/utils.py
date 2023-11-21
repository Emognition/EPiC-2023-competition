import pandas as pd
import os
from pathlib import Path
import numpy as np

def create_sliding_window_features(X, y, window_size=50):
    """
        X: DataFrame
        window_sizes: list of tuple [('col name', size), ..., ('col name', size)]
    """
    new_cols = [X.shift(i) for i in range(window_size+1)]
    new_X = pd.concat(new_cols, axis=1).dropna()
    start_index = np.intersect1d(y.index.tolist(), new_X.index.tolist()).min()

    return new_X.loc[y.loc[start_index:].index.tolist()], y.loc[start_index:]

def check_dir(*dirs):
    for d in dirs:
        if d.suffix != "":
            d = d.parent
        if not d.exists():
            d.mkdir(parents=True)


def data_splitter(load_dir, save_dir, loader, sub_vid_pairs, enable_sliding_window=False, window_size=50):
    check_dir(save_dir / 'train', save_dir / 'test')
    for sub, vid in sub_vid_pairs:
        X = pd.read_csv(os.path.join(load_dir, f'sub_{sub}_vid_{vid}.csv'), index_col='time')
        _, y = loader.train_data(sub, vid)
        
        if enable_sliding_window:
            X, y = create_sliding_window_features(X, y, window_size=50)

        df = pd.concat([X, y], axis=1)
        
        length = len(df)
        
        train_length = int(length * 0.8)
        # test_length = length - train_length

        df[:train_length].to_csv(save_dir / 'train' / f'sub_{sub}_vid_{vid}.csv', index_label='time')
        df[train_length:].to_csv(save_dir / 'test' / f'sub_{sub}_vid_{vid}.csv', index_label='time')


def fold_path(fold):
    return "fold_" + str(fold) if fold != -1 else ""
