from __future__ import annotations
import argparse
import neurokit2 as nk
import numpy as np
import os
import pandas as pd
import sys
sys.path.append(os.path.relpath("../src/"))
from dataloader import S1, S2, S3, S4
from dataset import EpicDataset
import torch
from torch.utils.data import DataLoader, Dataset
from sequential import SequenceEncoder

import pytorch_lightning as L
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger


torch.random.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Torch version: {torch.__version__}")
print(f"GPU?: {device}")

def run_train(args):
    seed_everything(1)
    checkpoint_dir = os.path.join(args.exp_dir, "checkpoints")
    checkpoint = ModelCheckpoint(
        checkpoint_dir,
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_weights_only=False,
        verbose=True,
    )
    train_checkpoint = ModelCheckpoint(
        checkpoint_dir,
        monitor="train_loss",
        mode="min",
        save_top_k=3,
        save_weights_only=False,
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [
        checkpoint,
        train_checkpoint,
        lr_monitor,
    ]
    sensorStr = args.sensor
    if len(args.sensor) > 1:
        sensorStr = '_'.join(args.sensor)
    if 'all' in args.sensor:
        sensorStr = 'all'
        args.sensor = [
            'ecg', 'bvp', 'gsr', 'rsp', 'skt',
            'emg_zygo', 'emg_coru', 'emg_trap'
        ]
    trainer = Trainer(
        default_root_dir=args.exp_dir,
        max_epochs=args.epochs,
        num_nodes=args.nodes,
        # gpus=args.gpus, # all?
        # accelerator="gpu",
#        strategy=DDPPlugin(find_unused_parameters=False),
        callbacks=callbacks,
        reload_dataloaders_every_n_epochs=1,
        gradient_clip_val=10.0,
        logger=TensorBoardLogger("tb_logs", name=f"LSTM_h{args.hidden}_e{args.epochs}_s{args.scenario}_o{args.offset}_{sensorStr}")
    )

    model = SequenceEncoder(args.numIn, args.hidden, args.out)
    data_module = DataLoader(
        getDatasetForScenario(args.scenario, args.sensor, args.offset),
        batch_size=64, shuffle=False, num_workers=8
    )
    trainer.fit(LitModel(model), data_module)#, ckpt_path=checkpoint_dir)
    trainer.save_checkpoint(f'model_h{args.hidden}_e{args.epochs}_s{args.scenario}_o{args.offset:02}_{sensorStr}.ckpt')


def getDatasetForScenario(n: int, modality: str, offset: int) -> Dataset:
    if n == 1:
        dataset = EpicDataset(S1(), modality, offset)
    elif n == 2:
        dataset = EpicDataset(S2(), modality, offset)
    elif n == 3:
        dataset = EpicDataset(S3(), modality, offset)
    else: #n == 4:
        dataset = EpicDataset(S4(), modality, offset)
    return dataset


class LitModel(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.save_hyperparameters()

    def training_step(self, batch):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.model(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, logger=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('numIn', type=int)
    parser.add_argument('hidden', type=int)
    parser.add_argument('out', type=int)
    parser.add_argument('sensor', type=str, nargs='+')
    parser.add_argument('-o', '--offset', type=int, default=0)
    parser.add_argument('-s', '--scenario', type=int, default=1)
    parser.add_argument('-d', '--exp_dir', type=str, default='MLexperiments')
    parser.add_argument('-e', '--epochs', default=5, type=int)
    parser.add_argument('-n', '--nodes', type=int, default=1)
    args = parser.parse_args()

    run_train(args)
