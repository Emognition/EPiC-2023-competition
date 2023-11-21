# train models for different scenarios
# usage: python3 train_models -s <1, 2, 3, 4> -n <num_gpus>
import multiprocessing
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append(os.path.relpath("../src/"))
import warnings

from dataloader import S1, S2, S3, S4
from train import train
from utils import check_dir

warnings.filterwarnings("ignore")
import argparse
import datetime
import logging


def train_for_scenario_1(num_gpus):
    s1 = S1(prefix)
    logging.info("start training models for scenario 1")
    for sub in s1.train_subs:
        for vid in s1.train_vids:
            X = pd.read_csv(input_path / f"sub_{sub}_vid_{vid}.csv", index_col="time")
            y = pd.read_csv(output_path / f"sub_{sub}_vid_{vid}.csv", index_col="time")
            train(
                X,
                y,
                "arousal",
                model_path / f"sub_{sub}_vid_{vid}_arousal",
                multiprocessing.cpu_count(),
                num_gpus,
            )
            logging.info(f"sub {sub} vid {vid} arousal fitted.")
            train(
                X,
                y,
                "valence",
                model_path / f"sub_{sub}_vid_{vid}_valence",
                multiprocessing.cpu_count(),
                num_gpus,
            )
            logging.info(f"sub {sub} vid {vid} valence fitted.")


def train_for_scenario_2(num_gpus):
    s2 = S2(prefix)
    logging.info("start training models for scenario 2")
    for fold in s2.fold:
        for vid in s2.train_vids[fold]:
            X = pd.read_csv(input_path / f"fold_{fold}_vid_{vid}.csv", index_col="time")
            y = pd.read_csv(
                output_path / f"fold_{fold}_vid_{vid}.csv", index_col="time"
            )
            train(
                X,
                y,
                "arousal",
                model_path / f"fold_{fold}_vid_{vid}_arousal",
                multiprocessing.cpu_count(),
                num_gpus,
            )
            logging.info(f"fold {fold} vid {vid} arousal fitted.")
            train(
                X,
                y,
                "valence",
                model_path / f"fold_{fold}_vid_{vid}_valence",
                multiprocessing.cpu_count(),
                num_gpus,
            )
            logging.info(f"fold {fold} vid {vid} valence fitted.")


def train_for_scenario_3(num_gpus):
    lst = [
        ("arousal", "arousal_03421"),
        ("valence", "valence_4102122"),
        ("arousal", "arousal_10162022"),
        ("valence", "valence_031620"),
    ]
    logging.info("start training models for scenario 3")
    for target, name in lst:
        X = pd.read_csv(input_path / f"{name}.csv", index_col="time")
        y = pd.read_csv(output_path / f"{name}.csv", index_col="time")
        train(
            X,
            y,
            target=target,
            model_path=model_path / f"{name}",
            num_cpus=multiprocessing.cpu_count(),
            num_gpus=num_gpus,
        )
        logging.info(f"{name} fitted.")


def train_for_scenario_4(num_gpus):
    s4 = S4()
    logging.info("start training models for scenario 4")
    for fold in s4.fold:
        for vid in s4.train_vids[fold]:
            X = pd.read_csv(input_path / f"vid_{vid}.csv", index_col="time")
            y = pd.read_csv(output_path / f"vid_{vid}.csv", index_col="time")
            train(
                X,
                y,
                "arousal",
                model_path / f"vid_{vid}_arousal",
                multiprocessing.cpu_count(),
                num_gpus,
            )
            logging.info(f"fold {fold} vid {vid} arousal fitted.")
            train(
                X,
                y,
                "valence",
                model_path / f"vid_{vid}_valence",
                multiprocessing.cpu_count(),
                num_gpus,
            )
            logging.info(f"fold {fold} vid {vid} valence fitted.")


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
    parser.add_argument("-n", "--num_gpus", type=int, required=True)

    args = parser.parse_args()
    scenario = args.scenario
    num_gpus = args.num_gpus

    assert scenario in [1, 2, 3, 4], logging.error(
        "scenario should be one of 1, 2, 3, 4"
    )
    prefix = "../"

    input_path = Path(prefix) / f"io_data/scenario_{scenario}" / "train" / "physiology"
    output_path = (
        Path(prefix) / f"io_data/scenario_{scenario}" / "train" / "annotations"
    )
    model_path = Path(prefix) / f"models/scenario_{scenario}"
    check_dir(model_path)

    if scenario == 1:
        train_for_scenario_1(num_gpus)
    elif scenario == 2:
        train_for_scenario_2(num_gpus)
    elif scenario == 3:
        train_for_scenario_3(num_gpus)
    else:
        train_for_scenario_4(num_gpus)
