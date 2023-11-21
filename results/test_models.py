# predict the valence and arousal using trained models
# usage: python3 test_models -s <1, 2, 3, 4>

import pandas as pd
from pathlib import Path

import sys, os
sys.path.append(os.path.relpath("../src/"))
from dataloader import S1, S2, S3, S4
from utils import check_dir
from test_model import test
import warnings
warnings.filterwarnings("ignore")
import logging, datetime

import argparse


def test_for_scenario_1():
    input_path = Path(prefix) / f'io_data/scenario_{scenario}' / 'test' / 'physiology'
    save_path = Path(prefix) / f'results/scenario_{scenario}/test/annotations'
    check_dir(save_path)
    s1 = S1(prefix)
    logging.info('start testing models for scenario 1')
    for sub in s1.test_subs:
        for vid in s1.test_subs:
            logging.info(f'start testing sub {sub} vid {vid}...')
            X = pd.read_csv(input_path / f'sub_{sub}_vid_{vid}.csv', index_col='time')
            test(X,
                 model_path / f'sub_{sub}_vid_{vid}_arousal',
                 model_path / f'sub_{sub}_vid_{vid}_valence',
                 save_path / f'sub_{sub}_vid_{vid}.csv',
                 )
            logging.info(f'testing sub {sub} vid {vid} finished.')


def test_for_scenario_2():
    s2 = S2(prefix)
    logging.info('start testing models for scenario 2')
    for fold in s2.fold:
        input_path = Path(prefix) / f'io_data/scenario_{scenario}/fold_{fold}/test/physiology'
        save_path = Path(prefix) / f'results/scenario_{scenario}/fold_{fold}/test/annotations'
        check_dir(save_path)
        for sub in s2.test_subs[fold]:
            for vid in s2.test_vids[fold]:
                X = pd.read_csv(input_path /f'sub_{sub}_vid_{vid}.csv', index_col='time')
                test(X,
                     model_path / f'fold_{fold}_vid_{vid}_arousal',
                     model_path / f'fold_{fold}_vid_{vid}_valence',
                     save_path / f'sub_{sub}_vid_{vid}.csv',
                     )
                logging.info(f'fold {fold} sub {sub} vid {vid} finished.')

def test_for_scenario_3():
    s3 = S3()
    for fold in s3.fold:
        save_path = Path(prefix) / f'results/scenario_3/fold_{fold}/test/annotations'
        input_path = Path(prefix) / f'io_data/scenario_3/fold_{fold}/test/physiology'
        check_dir(save_path)
        
        if fold == 0:
            arousal_model_path = model_path / 'arousal_03421'
            valence_model_path = model_path  / 'valence_4102122'
        elif fold == 1:
            arousal_model_path = model_path / 'arousal_10162022'
            valence_model_path = model_path  / 'valence_4102122'
        elif fold == 2:
            arousal_model_path = model_path / 'arousal_03421'
            valence_model_path = model_path / 'valence_031620'
        else:
            arousal_model_path = model_path / 'arousal_10162022'
            valence_model_path = model_path / 'valence_031620'
        
        for sub in s3.test_subs[fold]:
            for vid in s3.test_vids[fold]:
                logging.info(f'start testing fold {fold} sub {sub} ...')
                X = pd.read_csv(input_path / f'sub_{sub}_vid_{vid}.csv', index_col='time')
                test(X,
                     arousal_model_path,
                     valence_model_path,
                     model_path / f'fold_{fold}_vid_{vid}_valence',
                     save_path / f'sub_{sub}_vid_{vid}.csv',
                     )
                logging.info(f'finish predicting fold {fold} sub {sub} vid {vid}.')


def test_for_scenario_4():
    s4 = S4()
    logging.info('start testing models for scenario 4')
    for fold in s4.fold:
        save_path = Path(prefix) / f'results/scenario_4/fold_{fold}/test/annotations'
        input_path = Path(prefix) / f'io_data/scenario_4/fold_{fold}/test/physiology'
        check_dir(save_path)

        arousal_model_list = [model_path / f'vid_{vid}_arousal' for vid in s4.train_vids[fold]]
        valence_model_list = [model_path / f'vid_{vid}_valence' for vid in s4.train_vids[fold]]
        for sub in s4.test_subs[fold]:
            for vid in s4.test_vids[fold]:
                logging.info(f'start testing fold {fold} sub {sub} vid {vid} ...')
                X = pd.read_csv(input_path / f'sub_{sub}_vid_{vid}.csv', index_col='time')
                test(X,
                     arousal_model_list,
                     valence_model_list,
                     save_path / f'sub_{sub}_vid_{vid}.csv', late_fusion=True)
                logging.info(f'fold {fold} sub {sub} vid {vid} finished.')

if __name__ == '__main__':
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
    args = parser.parse_args()
    scenario = args.scenario

    assert scenario in [1, 2, 3, 4], logging.error('scenario should be one of 1, 2, 3, 4')
    prefix = '../'

    model_path = Path(prefix) / f'models/scenario_{scenario}'
    check_dir(model_path)

    if scenario == 1:
        test_for_scenario_1()
    elif scenario == 2:
        test_for_scenario_2()
    elif scenario == 3:
        test_for_scenario_3()
    else:
        test_for_scenario_4()
