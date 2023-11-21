"""
generate clean data and features.
    ../clean_data/scenario_x/<fold_x>/train or test/sub_x_vid_x.csv
    ../features/scenario_x/<fold_x>/train or test/sub_x_vid_x.csv

args:
    -s: scenario
    -f: fold,  (e.g., -f 1 -f 2)
"""

import sys, os
import multiprocessing
from functools import partial
import argparse
from pathlib import Path
sys.path.append(os.path.relpath("../src/"))
from dataloader import S2, S1, S3, S4
from utils import check_dir
from feature_extractor import feature_extractor

import logging, datetime

import warnings
warnings.filterwarnings("ignore")

log_format = '%(asctime)s [%(levelname)s] %(message)s'
log_filename = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
logging.basicConfig(format=log_format,
                    force=True,
                    handlers=[
                        logging.FileHandler(f"../log/{log_filename}.log"),
                        logging.StreamHandler()
                        ],
                    level=logging.INFO
                    )

parser = argparse.ArgumentParser(description='Process data, extract features')
parser.add_argument('-s', '--scenario', type=int, help='1, 2, 3, 4', required=True)
parser.add_argument('-f', '--fold', type=int, action='append')
parser.add_argument('-d', '--data', type=str, default='data', help="Name of the data folder")


def fold_path(fold):
    return "fold_" + str(fold) if fold != -1 else ""


def extractor(sub_vid, s, scenario, fold, train_test, data_path, feature_path):
    sub, vid = sub_vid
    if train_test == 'train':
        X, y = s.train_data(sub, vid) if fold == -1 else s.train_data(fold, sub, vid)
    else:
        X, y = s.test_data(sub, vid) if fold == -1 else s.test_data(fold, sub, vid)

    try:
        # skip when all exists
        if (feature_path / f'sub_{sub}_vid_{vid}.csv').exists() and (data_path / f'sub_{sub}_vid_{vid}.csv').exists():
            logging.info(f'scenario {scenario}{" fold " + str(fold) if fold != -1 else ""}: clean data and features exists')
        else:
            clean_data, features = feature_extractor(X, y)
            clean_data.to_csv(data_path / f'sub_{sub}_vid_{vid}.csv', index_label='time')
            features.to_csv(feature_path / f'sub_{sub}_vid_{vid}.csv', index_label='time')
            logging.info(f'scenario {scenario}{" fold " + str(fold) if fold != -1 else ""}: extracted features for {train_test} data (sub = {sub} vid = {vid}).')
    except Exception as e:
        logging.error(f'scenario {scenario}{" fold " + str(fold) if fold != -1 else ""} (sub = {sub} vid = {vid}): {e}')


if __name__ == "__main__":
    args = parser.parse_args()
    scenario = args.scenario

    assert scenario in [1, 2, 3, 4], logging.error('scenario should be one of 1, 2, 3, 4')

    root_path = Path('../')
    feature_path = root_path / 'features'
    clean_data_path = root_path / 'clean_data'

    if scenario == 1:
        s = S1(data=args.data)
    elif scenario == 2:
        s = S2(data=args.data)
    elif scenario == 3:
        s = S3(data=args.data)
    else:
        s = S4(data=args.data)

    folds = args.fold if args.fold is not None else s.fold

    logging.info(f'scenario {scenario}')

    for fold in folds:
        train_data_path = clean_data_path / f'scenario_{scenario}' / fold_path(fold) / 'train'
        test_data_path = clean_data_path / f'scenario_{scenario}' / fold_path(fold) / 'test'
        train_feature_path = feature_path / f'scenario_{scenario}' / fold_path(fold) / 'train'
        test_feature_path = feature_path / f'scenario_{scenario}' / fold_path(fold) / 'test'
        check_dir(clean_data_path,
                train_data_path,
                test_data_path,
                feature_path,
                train_feature_path,
                test_feature_path)

        func = partial(extractor, s=s, scenario=scenario, fold=fold,
                    train_test='train', data_path=train_data_path, feature_path=train_feature_path)
        pool_obj = multiprocessing.Pool()
        if (fold != -1):
            pool_obj.map(func, s.train_test_indices[fold]['train'])
        else:
            pool_obj.map(func, s.train_test_indices['train'])

        pool_obj.close()

        func = partial(extractor, s=s, scenario=scenario, fold=fold,
                    train_test='test', data_path=test_data_path, feature_path=test_feature_path)
        pool_obj = multiprocessing.Pool()
        if (fold != -1):
            pool_obj.map(func, s.train_test_indices[fold]['test'])
        else:
            pool_obj.map(func, s.train_test_indices['test'])
        pool_obj.close()