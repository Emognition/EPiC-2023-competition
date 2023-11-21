import torch
from torch.utils.data import Dataset
import numpy as np


class EpicDataset(Dataset):
    """Usage example:
    from dataloader import S1
    from dataset import EpicDataset
    trainScenario1Dataset = EpicDataset(S1())
    testScenario1Dataset = EpicDataset(S1(), is_test=True)
    """
    def __init__(
        self, scenario,
        modality:list=[
            'ecg', 'bvp', 'gsr', 'rsp', 'skt',
            'emg_zygo', 'emg_coru', 'emg_trap'
        ],
        offset:int=0, windowLen:int=50, is_test:bool=False
    ):
        if ('emg' in modality):
            if (len(modality) == 1) and (len(modality[0]) == 3):
                tmp = [x for x in modality if not 'emg' in x]
                tmp.extend(['emg_zygo', 'emg_coru', 'emg_trap'])
                modality = tmp

        self.is_test = is_test
        self.loader = scenario
        self.data = np.zeros((windowLen, 1, len(modality)))
        self.label = np.zeros((1, 2))
        self.numBlocks = 0
        for kk, (sub, vid) in enumerate([
            x[(not self.is_test)*'train'+self.is_test*'test']
            for x in scenario.train_test_groups()
        ]):
            data, label = self.loader.train_data(sub, vid)
            if (kk < 1):
                data = data[offset:]
            numBlocks = len(data) // windowLen
            endIdx = windowLen * numBlocks
            # rearrange data and annotations
            self.data = np.concatenate([
                self.data,
                data[:endIdx][modality].values.reshape(
                    windowLen, -1, len(modality)
                )
            ], axis=1)
            self.label = np.concatenate([
                self.label,
                label.values[len(label)-numBlocks:]
            ])
            self.numBlocks += numBlocks
        # transform to tensor
        self.data = torch.Tensor(self.data)
        self.label = torch.Tensor(self.label)

    def __len__(self):
        return self.numBlocks-1 # 1 dummy entry at the beginning to concatenate with 0's

    def __getitem__(self, idx):
        return self.data[:, 1+idx, :], self.label[1+idx, :]
