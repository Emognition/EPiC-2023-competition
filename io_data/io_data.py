# generate input and output for the models
# usage: python3 io_data.py -s <1, 2, 3, 4> -t <train or test>
#
# the input of test data in each scenario corresponds to one file (one sub and one vid)
# the input of train data in
#    scenario 1: one sub and one vid
#    scenario 2: eight per fold, each input corresponds to one video 30 subs
#    scenario 3: two input for arousal, two input for valence
#    scenario 4: four per fold, each input corresponds to one video

import os
import sys

sys.path.append(os.path.relpath("../src/"))
import multiprocessing
import warnings
from pathlib import Path

from dataloader import S1, S2, S3, S4
from io_generator import save_io
from utils import fold_path

warnings.filterwarnings("ignore")
import argparse
import datetime
import logging


def generate_io_for_scenario_1():
    s1 = S1(prefix)
    kwargs_list = []
    for fold in s1.fold:
        for sub in s1.subs(data_type):
            for vid in s1.vids(data_type):
                kwargs = dict()
                kwargs["scenario"] = scenario
                kwargs["input_path"] = (
                    Path(prefix)
                    / f"io_data/scenario_{scenario}"
                    / data_type
                    / "physiology"
                    / f"sub_{sub}_vid_{vid}.csv"
                )
                kwargs["output_path"] = (
                    Path(prefix)
                    / f"io_data/scenario_{scenario}"
                    / data_type
                    / "annotations"
                    / f"sub_{sub}_vid_{vid}.csv"
                )
                kwargs["past_window_size"] = past_window_size
                kwargs["future_window_size"] = future_window_size
                kwargs["prefix"] = prefix
                kwargs["fold_sub_vid_pairs"] = [(fold, sub, vid)]
                kwargs["data_type"] = data_type
                kwargs_list.append(kwargs)

    pool_obj = multiprocessing.Pool()
    pool_obj.map(save_io, kwargs_list)
    pool_obj.close()


def generate_io_for_scenario_2():
    s2 = S2(prefix)
    if data_type == "test":
        kwargs_list = []
        for fold in s2.fold:
            for sub in s2.subs(data_type)[fold]:
                for vid in s2.vids(data_type)[fold]:
                    kwargs = dict()
                    kwargs["scenario"] = scenario
                    kwargs["input_path"] = (
                        Path(prefix)
                        / f"io_data/scenario_{scenario}"
                        / fold_path(fold)
                        / data_type
                        / "physiology"
                        / f"sub_{sub}_vid_{vid}.csv"
                    )
                    kwargs["output_path"] = (
                        Path(prefix)
                        / f"io_data/scenario_{scenario}"
                        / fold_path(fold)
                        / data_type
                        / "annotations"
                        / f"sub_{sub}_vid_{vid}.csv"
                    )
                    kwargs["past_window_size"] = past_window_size
                    kwargs["future_window_size"] = future_window_size
                    kwargs["prefix"] = prefix
                    kwargs["fold_sub_vid_pairs"] = [(fold, sub, vid)]
                    kwargs["data_type"] = data_type
                    kwargs_list.append(kwargs)

        pool_obj = multiprocessing.Pool()
        pool_obj.map(save_io, kwargs_list)
        pool_obj.close()
        return

    kwargs_list = []
    for fold in s2.fold:
        for vid in s2.vids(data_type)[fold]:
            kwargs = dict()
            kwargs["scenario"] = scenario
            kwargs["input_path"] = (
                Path(prefix)
                / f"io_data/scenario_{scenario}"
                / data_type
                / "physiology"
                / f"fold_{fold}_vid_{vid}.csv"
            )
            kwargs["output_path"] = (
                Path(prefix)
                / f"io_data/scenario_{scenario}"
                / data_type
                / "annotations"
                / f"fold_{fold}_vid_{vid}.csv"
            )
            kwargs["past_window_size"] = past_window_size
            kwargs["future_window_size"] = future_window_size
            kwargs["prefix"] = prefix
            kwargs["data_type"] = data_type
            kwargs["fold_sub_vid_pairs"] = [
                (fold, sub, vid) for sub in s2.subs(data_type)[fold]
            ]
            kwargs_list.append(kwargs)

    pool_obj = multiprocessing.Pool()
    pool_obj.map(save_io, kwargs_list)
    pool_obj.close()


def generate_io_for_scenario_3():
    s3 = S3(prefix)
    if data_type == "test":
        kwargs_list = []
        for fold in s3.fold:
            for sub in s3.subs(data_type)[fold]:
                for vid in s3.vids(data_type)[fold]:
                    kwargs = dict()
                    kwargs["scenario"] = scenario
                    kwargs["input_path"] = (
                        Path(prefix)
                        / f"io_data/scenario_{scenario}"
                        / fold_path(fold)
                        / data_type
                        / "physiology"
                        / f"sub_{sub}_vid_{vid}.csv"
                    )
                    kwargs["output_path"] = (
                        Path(prefix)
                        / f"io_data/scenario_{scenario}"
                        / fold_path(fold)
                        / data_type
                        / "annotations"
                        / f"sub_{sub}_vid_{vid}.csv"
                    )
                    kwargs["past_window_size"] = past_window_size
                    kwargs["future_window_size"] = future_window_size
                    kwargs["prefix"] = prefix
                    kwargs["fold_sub_vid_pairs"] = [(fold, sub, vid)]
                    kwargs["data_type"] = data_type
                    kwargs_list.append(kwargs)

        pool_obj = multiprocessing.Pool()
        pool_obj.map(save_io, kwargs_list)
        pool_obj.close()
        return

    # arousal 03421
    kwargs_list = []
    kwargs = dict()
    kwargs["scenario"] = scenario
    kwargs["input_path"] = (
        Path(prefix)
        / f"io_data/scenario_{scenario}"
        / data_type
        / "physiology"
        / f"arousal_03421.csv"
    )
    kwargs["output_path"] = (
        Path(prefix)
        / f"io_data/scenario_{scenario}"
        / data_type
        / "annotations"
        / f"arousal_03421.csv"
    )
    kwargs["past_window_size"] = past_window_size
    kwargs["future_window_size"] = future_window_size
    kwargs["prefix"] = prefix
    kwargs["data_type"] = data_type
    kwargs["fold_sub_vid_pairs"] = [
        (0, sub, vid) for sub in s3.subs(data_type)[0] for vid in [0, 3, 4, 21]
    ]
    kwargs_list.append(kwargs)

    # valence 03421
    kwargs = dict()
    kwargs["scenario"] = scenario
    kwargs["input_path"] = (
        Path(prefix)
        / f"io_data/scenario_{scenario}"
        / data_type
        / "physiology"
        / f"valence_4102122.csv"
    )
    kwargs["output_path"] = (
        Path(prefix)
        / f"io_data/scenario_{scenario}"
        / data_type
        / "annotations"
        / f"valence_4102122.csv"
    )
    kwargs["past_window_size"] = past_window_size
    kwargs["future_window_size"] = future_window_size
    kwargs["prefix"] = prefix
    kwargs["data_type"] = data_type
    kwargs["fold_sub_vid_pairs"] = [
        (0, sub, vid) for sub in s3.subs(data_type)[0] for vid in [4, 10, 21, 22]
    ]
    kwargs_list.append(kwargs)

    # arousal 10162022
    kwargs = dict()
    kwargs["scenario"] = scenario
    kwargs["input_path"] = (
        Path(prefix)
        / f"io_data/scenario_{scenario}"
        / data_type
        / "physiology"
        / f"arousal_10162022.csv"
    )
    kwargs["output_path"] = (
        Path(prefix)
        / f"io_data/scenario_{scenario}"
        / data_type
        / "annotations"
        / f"arousal_10162022.csv"
    )
    kwargs["past_window_size"] = past_window_size
    kwargs["future_window_size"] = future_window_size
    kwargs["prefix"] = prefix
    kwargs["data_type"] = data_type
    kwargs["fold_sub_vid_pairs"] = [
        (3, sub, vid) for sub in s3.subs(data_type)[3] for vid in [10, 16, 20, 22]
    ]
    kwargs_list.append(kwargs)

    # arousal 10162022
    kwargs = dict()
    kwargs["scenario"] = scenario
    kwargs["input_path"] = (
        Path(prefix)
        / f"io_data/scenario_{scenario}"
        / data_type
        / "physiology"
        / f"valence_031620.csv"
    )
    kwargs["output_path"] = (
        Path(prefix)
        / f"io_data/scenario_{scenario}"
        / data_type
        / "annotations"
        / f"valence_031620.csv"
    )
    kwargs["past_window_size"] = past_window_size
    kwargs["future_window_size"] = future_window_size
    kwargs["prefix"] = prefix
    kwargs["data_type"] = data_type
    kwargs["fold_sub_vid_pairs"] = [
        (3, sub, vid) for sub in s3.subs(data_type)[3] for vid in [0, 3, 16, 20]
    ]
    kwargs_list.append(kwargs)

    pool_obj = multiprocessing.Pool()
    pool_obj.map(save_io, kwargs_list)
    pool_obj.close()


def generate_io_for_scenario_4():
    s4 = S4(prefix)
    if data_type == "test":
        kwargs_list = []
        for fold in s4.fold:
            for sub in s4.subs(data_type)[fold]:
                for vid in s4.vids(data_type)[fold]:
                    kwargs = dict()
                    kwargs["scenario"] = scenario
                    kwargs["input_path"] = (
                        Path(prefix)
                        / f"io_data/scenario_{scenario}"
                        / fold_path(fold)
                        / data_type
                        / "physiology"
                        / f"sub_{sub}_vid_{vid}.csv"
                    )
                    kwargs["output_path"] = (
                        Path(prefix)
                        / f"io_data/scenario_{scenario}"
                        / fold_path(fold)
                        / data_type
                        / "annotations"
                        / f"sub_{sub}_vid_{vid}.csv"
                    )
                    kwargs["past_window_size"] = past_window_size
                    kwargs["future_window_size"] = future_window_size
                    kwargs["prefix"] = prefix
                    kwargs["fold_sub_vid_pairs"] = [(fold, sub, vid)]
                    kwargs["data_type"] = data_type
                    kwargs_list.append(kwargs)

        pool_obj = multiprocessing.Pool()
        pool_obj.map(save_io, kwargs_list)
        pool_obj.close()
        return

    kwargs_list = []
    for fold in s4.fold:
        for vid in s4.vids(data_type)[fold]:
            kwargs = dict()
            kwargs["scenario"] = scenario
            kwargs["input_path"] = (
                Path(prefix)
                / f"io_data/scenario_{scenario}"
                / data_type
                / "physiology"
                / f"vid_{vid}.csv"
            )
            kwargs["output_path"] = (
                Path(prefix)
                / f"io_data/scenario_{scenario}"
                / data_type
                / "annotations"
                / f"vid_{vid}.csv"
            )
            kwargs["past_window_size"] = past_window_size
            kwargs["future_window_size"] = future_window_size
            kwargs["prefix"] = prefix
            kwargs["data_type"] = data_type
            kwargs["fold_sub_vid_pairs"] = [
                (fold, sub, vid) for sub in s4.subs(data_type)[fold]
            ]
            kwargs_list.append(kwargs)

    pool_obj = multiprocessing.Pool()
    pool_obj.map(save_io, kwargs_list)
    pool_obj.close()


if __name__ == "__main__":
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    log_filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logging.basicConfig(
        format=log_format,
        force=True,
        handlers=[
            logging.FileHandler(f"../log/{log_filename}.log"),
            logging.StreamHandler(),
        ],
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser(description="Process data, extract features")
    parser.add_argument("-s", "--scenario", type=int, help="1, 2, 3, 4", required=True)
    parser.add_argument("-t", "--type", type=str, help="train or test", required=True)

    args = parser.parse_args()
    scenario = args.scenario
    data_type = args.type

    assert scenario in [1, 2, 3, 4], logging.error(
        "scenario should be one of 1, 2, 3, 4"
    )
    assert data_type in ["train", "test"], logging.error("type should be train or test")

    prefix = "../"
    past_window_size = 50
    future_window_size = 50

    if scenario == 1:
        generate_io_for_scenario_1()
    elif scenario == 2:
        generate_io_for_scenario_2()
    elif scenario == 3:
        generate_io_for_scenario_3()
    else:
        generate_io_for_scenario_4()
