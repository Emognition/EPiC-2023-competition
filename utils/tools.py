import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj =='type3':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == 'type4':
        lr_adjust = {epoch: args.learning_rate * (0.9 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type5':
        lr_adjust = {
            2: 5e-4, 4: 3e-4, 6: 1e-4, 8: 5e-5,
            10: 3e-5, 12:1e-5,14:5e-6,16:1e-6,18:5e-7,20:3e-7,22:1e-7,24:5e-8,26:3e-8,28:1e-8 
            # 12: 1e-5, 14: 5e-6, 16:3e-6, 18:1e-6,
            # 20: 5e-7,22: 3e-7, 24: 1e-7, 26:5e-8, 28:1e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7,verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        # self.accelerator= accelerator
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, state, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, state, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, state, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, state, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
   
        torch.save(state, path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def load_data_no_folds(scenario_dir_path, dataset_type):
    # make dict to store data
        storage_list = list()
        # make paths for the specified dataset
        train_annotations_dir = Path(scenario_dir_path, dataset_type, "annotations")
        train_physiology_dir = Path(scenario_dir_path, dataset_type, "physiology")
        # sort contents of dirs, so that physiology and annotations are in the same order  
        train_physiology_files = sorted(Path(train_physiology_dir).iterdir())
        train_annotation_files = sorted(Path(train_annotations_dir).iterdir())
        # iterate over annotation and physiology files

        storage_list = [(annotations_file_path.name, pd.read_csv(physiology_file_path), pd.read_csv(annotations_file_path))\
                    for physiology_file_path, annotations_file_path in zip(train_physiology_files, train_annotation_files)]
        # for physiology_file_path, annotations_file_path in zip(train_physiology_files, train_annotation_files):
        #     # make sure that we load corresponding physiology and annotations
        #     assert physiology_file_path.name == annotations_file_path.name, "Order mismatch"
        #     # load data from files
        #     df_physiology = pd.read_csv(physiology_file_path)
        #     df_annotations = pd.read_csv(annotations_file_path)
        #     # continue # comment / delete this line if you want to store data in data_store list
        #     # store data
        #     storage_list.append((annotations_file_path.name, df_physiology, df_annotations))
        return storage_list
    
def load_data_with_folds(scenario_dir_path, dataset_type):
    # make dict to store data
        storage_dict = dict()
        # iterate over the scenario directory
        for fold_dir in Path(scenario_dir_path).iterdir():
            # make paths for current fold
            train_annotations_dir = Path(fold_dir, f"{dataset_type}/annotations/")
            train_physiology_dir = Path(fold_dir, f"{dataset_type}/physiology/")
            # make key in a dict for current fold 
            storage_dict.setdefault(fold_dir.name, list())
            # sort contents of dirs, so that physiology and annotations are in the same order  
            train_physiology_files = sorted(Path(train_physiology_dir).iterdir())
            train_annotation_files = sorted(Path(train_annotations_dir).iterdir())
            # iterate over annotation and physiology files
            for physiology_file_path, annotations_file_path in zip(train_physiology_files, train_annotation_files):
                # make sure that we load corresponding physiology and annotations
                assert physiology_file_path.name == annotations_file_path.name, "Order mismatch"
                # load data from files
                df_physiology = pd.read_csv(physiology_file_path)
                df_annotations = pd.read_csv(annotations_file_path)
                # continue # comment / delete this line if you want to store data in data_store list
                # store data
                storage_dict[fold_dir.name].append((annotations_file_path.name, df_physiology, df_annotations))
        return storage_dict
