import argparse
import torch
import torch.nn as nn
import numpy as np
import os, sys
import pandas as pd
import pytorch_lightning as L
from pytorch_lightning import Trainer
sys.path.append(os.path.relpath("../src/"))
from sequential import SequenceEncoder
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


class LitModel(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=['model'])

    def training_step(self, batch):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.model(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def forward(self, data):
        x = data.flatten()
        x = x.unsqueeze(0)
        x = torch.Tensor.to(x)
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)


def run_test(args) -> float:
    windowLen = 50
    numSensors = len(args.modality)
    model = LitModel(SequenceEncoder(windowLen*numSensors, 128, 2))
    model.load_from_checkpoint(checkpoint_path=args.model)
    model.eval()

    df_data = pd.read_csv(args.data)

    overallLoss = np.zeros(len(df_data))
    for file in tqdm(df_data.filename.values):
      fileLoss = 0
      filepath = '../data/scenario_3/fold_1/train'
      data_filename = os.path.join(filepath, 'physiology', file)
      label_filename = os.path.join(filepath, 'annotations', file)

      df_x = pd.read_csv(data_filename, index_col='time')
      df_y = pd.read_csv(label_filename, index_col='time')

      numBlocks = len(df_x) // windowLen
      endIdx = windowLen * numBlocks

      x = torch.Tensor(df_x[:endIdx][args.modality].values.reshape(windowLen, -1, numSensors))
      y = torch.Tensor(df_y.values)
      y_hat = torch.zeros((x.shape[1], 2))
      for mm in range(x.shape[1]):
          y_hat[mm, :] = model(x[:, mm, :])
      overallLoss[kk] = torch.sqrt(nn.functional.mse_loss(y_hat, y[1:,:]))
    return np.mean(overallLoss)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('sensor', type=str)
    parser.add_argument('data', type=str)
    parser.add_argument(
        '-m', '--modality', type=str, nargs='+',
        default=['ecg', 'bvp', 'gsr', 'rsp', 'skt', 'emg_zygo', 'emg_coru', 'emg_trap']
    )
    args = parser.parse_args()

    models = sorted([x for x in os.listdir('.') if x.endswith(f'{args.sensor}.ckpt')])
    offset = [int(x.split('_')[4][1:]) for x in models]

    rmse = np.zeros(len(models))
    for kk, model in enumerate(models):
        print(f'Evaluate model {kk+1}/{len(models)}: {model}')
        args.model = model
        rmse[kk] = run_test(args)
        print(f'RMSE ({args.sensor} {offset[kk]}): {rmse[kk]}')

    df = pd.DataFrame(
        np.array([
            offset,
            rmse
        ]).transpose(), columns=['offset', 'rmse']
    )
    df.to_csv(f'performance_{args.sensor}.csv')