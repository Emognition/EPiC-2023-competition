"""
plot train data features.
    ../features/figures/scenario_x/<fold_x>/train or test/sub_x_vid_x.csv

args:
    -s: scenario
    -f: fold,  (e.g., -f 1 -f 2)
"""

import sys, os 
import multiprocessing
from functools import partial
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
sys.path.append(os.path.relpath("../src/"))
from dataloader import S2, S1, S3, S4
from utils import check_dir, fold_path

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings("ignore")

plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.8
plt.rcParams['grid.linestyle'] = 'dotted'
# plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams['figure.figsize'] = (4.845, 3.135)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['mathtext.default']='regular'

colors = ListedColormap(sns.color_palette('deep')).colors
arousal_color = colors[0]
valence_color = colors[1]
bg_color = ListedColormap(sns.color_palette('Greys', 4)).colors[0]
feature_color = ListedColormap(sns.color_palette('Greens', 4)).colors[3]

parser = argparse.ArgumentParser(description='Process data, extract features')
parser.add_argument('-s', '--scenario', type=int, help='1, 2, 3, 4', required=True)
parser.add_argument('-f', '--fold', type=int, action='append')
parser.add_argument('--sub', type=int)
parser.add_argument('--vid', type=int)

args = parser.parse_args()
scenario = args.scenario

assert scenario in [1, 2, 3, 4], 'scenario should be one of 1, 2, 3, 4'


if scenario == 1:
    s = S1()
elif scenario == 2:
    s = S2()
elif scenario == 3:
    s = S3()
else: 
    s = S4()

folds = args.fold if args.fold is not None else s.fold

root_path = Path('../')


def plot_features(scenario, fold, sub, vid):
    feature_path = Path(f'scenario_{scenario}') / fold_path(fold) / 'train' / f'sub_{sub}_vid_{vid}.csv'
    features = pd.read_csv(feature_path, index_col='time').fillna(0)
    _, y = s.train_data(sub, vid) if fold == -1 else s.train_data(fold, sub, vid)
    width_per_figure = 4
    height_per_figure = 2
    n_row = 8
    n_column = 9
    fig, axes = plt.subplots(n_row, n_column, figsize=(width_per_figure * n_column, height_per_figure * n_row))
    axes = axes.flatten()
    X = features.index
    for i, col in enumerate(features.columns):
        axes[i].set_ylabel(f'{col}')
        axes[i].plot(X.to_numpy(), features[col].to_numpy(), color=feature_color, linewidth=0.8)
        ax2 = plt.twinx(axes[i])
        ax2.plot(y['arousal'], label='arousal', color=arousal_color, linewidth=1.2)
        ax2.plot(y['valence'], label='valence', color=valence_color, linewidth=1.2)
        plt.legend()
    # plt.suptitle()
    plt.tight_layout()

    save_path = feature_path / 'figures' / f'scenario_{scenario}'/ fold_path(fold) / 'train'
    check_dir(save_path)
    plt.savefig(save_path / f'sub_{sub}_vid_{vid}.pdf', dpi=300, format='pdf')
    plt.close()

def plot_features_for_scenario(scenario, folds):
    for fold in folds:
        for sub in s.train_subs if fold == -1 else s.train_subs[fold]:
            for vid in s.train_vids if fold == -1 else s.train_vids[fold]:
                plot_features(scenario, fold, sub, vid)

def plot_features_for_sub_vid(scenario, fold, sub, vid):
    plot_features(scenario, fold, sub, vid)

if args.sub is not None and args.vid is not None:
    plot_features_for_sub_vid(scenario, folds[0], args.sub, args.vid)

else:
    plot_features(scenario, folds)
