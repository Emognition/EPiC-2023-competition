# to use the dataloader, in the notebook, run the following code
# sys.path.append(os.path.relpath("../src/"))
# from dataloader import S1, S2, S3, S4

# s1 = S1()
# s2 = S2()
# s3 = S3()
# s4 = S4()

# """
#     property:
#         train_subs, train_vids, test_subs, test_vids, fold, scenario, keys,
#         update Apr 21: train_test_indices
#     function:
#         S1:
#           train_data(sub, vid, <feature(s)>) -> tuple (X, y)
#           test_data(sub, vid, <feature(s)>) -> tuple (X, y)
#         S2, 3, 4
#           train_data(fold, sub, vid, <feature(s)>) -> tuple (X, y)
#           test_data(fold, sub, vid, <feature(s)>) -> tuple (X, y)
#        'feature(s)' is an optinal parameter, str or list of str
#
#         return list of train_test pairs for each model.
#         train_test_groups(...) -> list(dict('train': [(sub, vid), ..., (sub, vid)], 'test': [(sub, vid), ..., (sub, vid)]))
# """

import os
import re
from functools import partial
from pathlib import Path

import pandas as pd


class BaseLoader:
    def __init__(self, scenario: int, fold: list, path_prefix="../", data="data"):
        """
        property:
            train_subs, train_vids, test_subs, test_vids
        function:
            train_data(...),
            test_data(...),
            train_test_groups(...)
        """
        self.scenario = scenario
        self.fold = fold if fold != [] else [-1]
        self.path_prefix = path_prefix
        self.data_path = data
        self.keys = [
            "ecg",
            "bvp",
            "gsr",
            "rsp",
            "skt",
            "emg_zygo",
            "emg_coru",
            "emg_trap",
        ]
        (
            self.train_subs,
            self.train_vids,
            self.test_subs,
            self.test_vids,
        ) = self._initialization()

        self.train_data = (
            self._train_data if self.fold[0] != -1 else partial(self._train_data, -1)
        )
        self.test_data = (
            self._test_data if self.fold[0] != -1 else partial(self._test_data, -1)
        )
        self.train_test_indices = (
            self._train_test_indices_with_fold
            if self.fold[0] != -1
            else self._train_test_indices_without_fold
        )

    def _initialization(self):
        """
        collect subs and vids
        """
        train_subs = []
        train_vids = []
        test_subs = []
        test_vids = []
        for i in self.fold:
            filenames = os.listdir(
                Path(self.path_prefix)
                / f"{self.data_path}/scenario_{self.scenario}"
                / f'{"fold_" + str(i) if i != -1 else ""}'
                / "train/physiology"
            )
            train_subs.append(
                sorted(
                    list(
                        set([int(re.findall(r"(?<=sub_)\d+", s)[0]) for s in filenames])
                    )
                )
            )
            train_vids.append(
                sorted(
                    list(
                        set([int(re.findall(r"(?<=vid_)\d+", s)[0]) for s in filenames])
                    )
                )
            )

            filenames = os.listdir(
                Path(self.path_prefix)
                / f"{self.data_path}/scenario_{self.scenario}"
                / f'{"fold_" + str(i) if i != -1 else ""}'
                / "test/physiology"
            )
            test_subs.append(
                sorted(
                    list(
                        set([int(re.findall(r"(?<=sub_)\d+", s)[0]) for s in filenames])
                    )
                )
            )
            test_vids.append(
                sorted(
                    list(
                        set([int(re.findall(r"(?<=vid_)\d+", s)[0]) for s in filenames])
                    )
                )
            )

        if self.fold[0] != -1:
            return train_subs, train_vids, test_subs, test_vids
        else:
            return train_subs[0], train_vids[0], test_subs[0], test_vids[0]

    def _load_data(self, prefix, sub, vid, features):
        features = self._check_features(features)
        return pd.read_csv(
            os.path.join(
                self.path_prefix, prefix, "physiology", f"sub_{sub}_vid_{vid}.csv"
            ),
            index_col="time",
        )[features], pd.read_csv(
            os.path.join(
                self.path_prefix, prefix, "annotations", f"sub_{sub}_vid_{vid}.csv"
            ),
            index_col="time",
        )

    def _check_features(self, features):
        assert (
            type(features) == str or type(features) == list
        ), "The type of <features> should be str or list"

        if type(features) == str:
            features = [features]
        else:
            if len(features) == 0:
                features = self.keys
        for feature in features:
            assert feature in self.keys, f"{feature} is not a feature"

        return features

    def _train_data(self, fold: int, sub: int, vid: int, features=[]):
        """
        features: list or str
        return X, y
        """
        return self._load_data(
            Path(f"{self.data_path}/scenario_{self.scenario}")
            / f'{"fold_" + str(fold) if fold != -1 else ""}'
            / "train",
            sub,
            vid,
            features,
        )

    def _test_data(self, fold: int, sub: int, vid: int, features=[]):
        """
        features: list or str
        return X, y
        """
        return self._load_data(
            Path(f"{self.data_path}/scenario_{self.scenario}")
            / f'{"fold_" + str(fold) if fold != -1 else ""}'
            / "test",
            sub,
            vid,
            features,
        )

    @property
    def _train_test_indices_without_fold(self):
        """
        return dict('train', 'test')
        """
        return {
            "train": [(sub, vid) for vid in self.train_vids for sub in self.train_subs],
            "test": [(sub, vid) for vid in self.test_vids for sub in self.test_subs],
        }

    @property
    def _train_test_indices_with_fold(self):
        """
        return dict('train', 'test')
        """
        return [
            {
                "train": [
                    (sub, vid)
                    for vid in self.train_vids[i]
                    for sub in self.train_subs[i]
                ],
                "test": [
                    (sub, vid) for vid in self.test_vids[i] for sub in self.test_subs[i]
                ],
            }
            for i in self.fold
        ]

    def subs(self, type):
        if type == "train":
            return self.train_subs
        elif type == "test":
            return self.test_subs
        else:
            raise Exception("Type should be train or test.")

    def vids(self, type):
        if type == "train":
            return self.train_vids
        elif type == "test":
            return self.test_vids
        else:
            raise Exception("Type should be train or test.")


class S1(BaseLoader):
    def __init__(self, path_prefix="../", data="data"):
        super().__init__(1, [], path_prefix, data)

    def train_test_groups(self):
        return [
            {"train": (sub, vid), "test": (sub, vid)}
            for vid in self.train_vids
            for sub in self.train_subs
        ]


class S2(BaseLoader):
    def __init__(self, path_prefix="../", data="data"):
        super().__init__(2, [0, 1, 2, 3, 4], path_prefix, data)

    def train_test_groups(self, same_vids=True):
        """

        if "same vids", different subs, same vid -> different subs, same vid
        otherwise, different subs, different vids -> different subs, different vids
        """
        if same_vids:
            return [
                [
                    {
                        "train": [(sub, vid) for sub in self.train_subs[i]],
                        "test": [(sub, vid) for sub in self.test_subs[i]],
                    }
                    for vid in self.train_vids[i]
                ]
                for i in self.fold
            ]
        else:
            return [
                {
                    "train": [
                        (sub, vid)
                        for sub in self.train_subs[i]
                        for vid in self.train_vids[i]
                    ],
                    "test": [
                        (sub, vid)
                        for sub in self.test_subs[i]
                        for vid in self.train_vids[i]
                    ],
                }
                for i in self.fold
            ]


class S3(BaseLoader):
    def __init__(self, path_prefix="../", data="data"):
        super().__init__(3, [0, 1, 2, 3], path_prefix, data)

    def train_test_groups(self, same_subs=True):
        """
        if "same subs", same sub but different vids -> same sub, different vids
        otherwise, different subs, different vids -> different subs, different vids

        return
            [ # fold
                {
                    'train': (sub, vid), ... (sub, vid),
                    'test': (sub, vid), ..., (sub, vid))
                },
                {...}
            ]
        """
        if same_subs:
            return [
                [
                    {
                        "train": [(sub, vid) for vid in self.train_vids[i]],
                        "test": [(sub, vid) for vid in self.test_vids[i]],
                    }
                    for sub in self.train_subs[i]
                ]
                for i in self.fold
            ]
        else:
            return [
                {
                    "train": [
                        (sub, vid)
                        for sub in self.train_subs[i]
                        for vid in self.train_vids[i]
                    ],
                    "test": [
                        (sub, vid)
                        for sub in self.test_subs[i]
                        for vid in self.train_vids[i]
                    ],
                }
                for i in self.fold
            ]


class S4(BaseLoader):
    def __init__(self, path_prefix="../", data="data"):
        super().__init__(4, [0, 1], path_prefix, data)

    def train_test_groups(self):
        pass
