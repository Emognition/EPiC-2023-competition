import numpy as np
import pandas as pd
import os
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import itertools
from itertools import combinations, product
import tqdm
import glob
import os.path
import sys
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.tools import EarlyStopping,load_data_no_folds,load_data_with_folds
import matplotlib.pyplot as plt
from pathlib import Path
import copy
import argparse

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='File level Prediction Smoothing')

parser.add_argument('--scenario', type=int, default= 1,
                    help='data scenario, options:[1,2,3,4],')
parser.add_argument('--vali_fold', type=int, default=0,
                    help='which fold to be validation options:[0,1,2,3,4]')
parser.add_argument('--label_len', type=int, default=1, help='start token length')
parser.add_argument('--pred_len', type=int, default=10, help='prediction sequence length')
parser.add_argument('--root_path', type=str, default='/data/EPiC_project/EPiC-2023-competition-data-v1.0/data', help='root path of the data file')
args = parser.parse_args()

if args.scenario==1:
    folder_path_train = f"./features/scenario_{args.scenario}/fold_0_ll_{args.label_len}"
else:
    folder_path_train = f"./features/scenario_{args.scenario}/fold_{args.vali_fold}_ll_{args.label_len}"

fn_dir = f"{folder_path_train}/file_name" 
enc_out_dir = f"{folder_path_train}/enc_out" 
out_dir = f"{folder_path_train}/outputs" 
lab_dir = f"{folder_path_train}/labels" 

train_out = np.load(f"{out_dir}/train_out.npy")
vali_out = np.load(f"{out_dir}/vali_out.npy")
test_out = np.load(f"{out_dir}/test_out.npy")
train_lab = np.load(f"{lab_dir}/train_lab.npy")
vali_lab = np.load(f"{lab_dir}/vali_lab.npy")
test_lab = np.load(f"{lab_dir}/test_lab.npy")
train_file = np.load(f"{fn_dir}/train_file.npy")
vali_file = np.load(f"{fn_dir}/valdation_file.npy")
test_file = np.load(f"{fn_dir}/test_file.npy")



scenario_dir = f"{args.root_path}/scenario_{args.scenario}"

if args.scenario == 1:
    storage = load_data_no_folds(scenario_dir,"train")
else:
    storage = load_data_with_folds(scenario_dir,"train")


if args.scenario == 1:
    pass
else:
    num_validation = len(storage[f"fold_{args.vali_fold}"])//2
    fold_list = list(storage.keys())
    train_fold = [fl for fl in fold_list if fl != f"fold_{args.vali_fold}"]
overlap = args.label_len-1+ args.pred_len
lab_pred_len = args.label_len-1+ args.pred_len

if args.scenario == 1:
    fn = [fn for fn ,_,_ in storage]
    train_file_lv_all = [(train_out[train_file==f],train_lab[train_file==f]) for f in fn]
    vali_file_lv_all = [(vali_out[vali_file==f],vali_lab[vali_file==f]) for f in fn]
    test_file_lv_all = [(test_out[test_file==f],test_lab[test_file==f]) for f in fn]

else:
    sh = [tr.shape[0] for tr in train_file]
    itv = [sum(sh[:i]) for i in range(len(sh)+1)]
    itvs = [(itv[i-1],itv[i]) for i in range(1,len(itv))]
    val_fn = [fn for fn,_,_ in storage[f"fold_{args.vali_fold}"][:num_validation]]
    test_fn = [fn for fn,_,_ in storage[f"fold_{args.vali_fold}"][num_validation:]]
    train_file_lv_all = [(train_out[it[0]:it[1]],train_lab[it[0]:it[1]]) for it in itvs]
    vali_file_lv_all = [(vali_out[vali_file==f],vali_lab[vali_file==f]) for f in val_fn]
    est_file_lv_all = [(test_out[test_file==f],test_lab[test_file==f]) for f in test_fn]

print(train_file_lv_all[0][0].shape,train_file_lv_all[0][-1].shape,\
      vali_file_lv_all[0][0].shape,vali_file_lv_all[0][-1].shape,\
      test_file_lv_all[0][0].shape,test_file_lv_all[0][-1].shape)

class EPiCDataset(Dataset):
    def __init__(self, train_file_all, vali_file_all,test_file_all,task="smooth",flag="train"):
        self.train_file_all = train_file_all
        self.vali_file_all = vali_file_all
        self.test_file_all = test_file_all
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.task = task
        self.__read_data__()

    def __read_data__(self):
        
        if self.task == "smooth":
            concat_len = 10
            overlap = 5
            train_feature = [np.array([np.vstack(pred[i-concat_len:i]) for i in range(concat_len,pred.shape[0],overlap)])\
                             for pred,lab in self.train_file_all]
            vali_feature = [np.array([np.vstack(pred[i-concat_len:i]) for i in range(concat_len,pred.shape[0],overlap)])\
                             for pred,lab in self.vali_file_all]
            test_feature = [np.array([np.vstack(pred[i-concat_len:i]) for i in range(concat_len,pred.shape[0],overlap)])\
                             for pred,lab in self.test_file_all]
            
            train_label = [np.array([np.vstack(lab[i-concat_len:i]) for i in range(concat_len,lab.shape[0],overlap)])\
                             for pred,lab in self.train_file_all]
            vali_label = [np.array([np.vstack(lab[i-concat_len:i]) for i in range(concat_len,lab.shape[0],overlap)])\
                             for pred,lab in self.vali_file_all]
            test_label = [np.array([np.vstack(lab[i-concat_len:i]) for i in range(concat_len,lab.shape[0],overlap)])\
                             for pred,lab in self.test_file_all]
            
            train_feature = np.vstack(train_feature)
            vali_feature = np.vstack(vali_feature)
            test_feature = np.vstack(test_feature)
            train_label = np.vstack(train_label)
            vali_label = np.vstack(vali_label)
            test_label = np.vstack(test_label)
            
            train_size = train_feature.shape[0]
            valdation_size = vali_feature.shape[0]
            test_size = test_feature.shape[0]

            feature_all = np.vstack([train_feature,vali_feature,test_feature])
            label_all = np.vstack([train_label,vali_label,test_label])

            border1s = [0, train_size, feature_all.shape[0]-test_size]
            border2s = [train_size, train_size + valdation_size, feature_all.shape[0]]

            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]


            self.data_x = feature_all[border1:border2]
            self.data_y = label_all[border1:border2]
            
            
            
    
    def __getitem__(self, idx):
        return self.data_x[idx], self.data_y[idx]
    
    def __len__(self):
        
        return len(self.data_x)
    
class ConV1dSmooth(nn.Module):
    
    def __init__(self,feature_size,hidden_size):
        super(ConV1dSmooth, self).__init__()

        self.conv1 = nn.Conv1d(feature_size, hidden_size*2, 5)
        self.conv2 = nn.Conv1d(hidden_size*2, hidden_size*2, 5,padding=2)
        self.conv3 = nn.Conv1d(hidden_size*2, hidden_size*2, 5,padding=4)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.fc_e = nn.Linear(hidden_size*2, hidden_size)
        self.fc_b = nn.Linear(hidden_size, hidden_size//2)
        self.fc_o = nn.Linear(hidden_size//2, 2)


    def forward(self, x):
        
        x = x.transpose(1,2)
        x = self.conv1(x) 
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = x.transpose(1,2)
        x_v = self.dropout(self.fc_e(x))
        x_v = self.relu(x_v)
        x_v = self.dropout(self.fc_b(x_v))
        x_v = self.relu(x_v)
        x_v = self.dropout(self.fc_o(x_v))

        return x_v
    
def val(model, val_loader, criterion, device):
    total_loss = []
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(val_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            out = model(batch_x)
            out = out.detach().cpu()
            batch_y = batch_y.detach().cpu()
            loss = criterion(out,batch_y)
            
            total_loss.append(loss)
    total_loss = np.average(total_loss)
    model.train()
    return total_loss

def adjust_learning_rate(optimizer,learning_rate, epoch, lradj):
    if lradj == 'type1':
        lr_adjust = {epoch: learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif lradj == 'type2':
        lr_adjust = {
            10: 5e-5, 20: 3e-5, 30: 1e-5,
        }
    elif lradj =='type3':
        lr_adjust = {epoch: learning_rate}
    elif lradj == 'type4':
        lr_adjust = {epoch: learning_rate * (0.9 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

def test(model,test_loader, device):
        
        inps = []
        preds = []
        trues = []

        model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                out = model(batch_x)
                inp = batch_x.detach().cpu().numpy()
                pred = out.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                inps.append(inp)
                preds.append(pred)
                trues.append(true)
        inps = np.vstack(inps)
        preds = np.vstack(preds)
        trues = np.vstack(trues)
        print('test shape:',inps.shape, preds.shape, trues.shape)

        return inps,preds,trues

def predict(model,test_tensor, device):
 
        model.eval()
        with torch.no_grad():
            batch_x = test_tensor.float().to(device)
            out = model(batch_x)
            inp = batch_x.detach().cpu().numpy()
            pred = out.detach().cpu().numpy()
        print('test shape:',inp.shape, pred.shape)

        return inp,pred


def train(model, train_loader, val_loader,test_loader,epochs, learning_rate,device):
    
        path = f"./smooth_checkpoints/sc_{args.scenario}_fold_{args.vali_fold}"
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=10, verbose=True)
        time_now = time.time()

        train_steps = len(train_loader)

        model_optim = optim.Adam(model.parameters(),lr=learning_rate)
        criterion = nn.SmoothL1Loss(beta=1.35)

        for epoch in range(epochs):
            iter_count = 0
            train_loss = []

            model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                out = model(batch_x)
                loss = criterion(out, batch_y)
                train_loss.append(loss.item())
                
                if (i + 1) % 10 == 0:
                    # print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((epochs - epoch) * train_steps - i)
                    # print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = val(model,val_loader, criterion,device)
            test_loss = val(model,test_loader, criterion,device)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, learning_rate ,epoch + 1, "type2")


train_dataset = EPiCDataset(train_file_lv_all,vali_file_lv_all,test_file_lv_all,task="smooth",flag="train")
val_dataset = EPiCDataset(train_file_lv_all,vali_file_lv_all,test_file_lv_all,task="smooth",flag="val")
test_dataset = EPiCDataset(train_file_lv_all,vali_file_lv_all,test_file_lv_all,task="smooth",flag="test")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

feature_size = 2
hidden_size = 512
num_epochs = 30
learning_rate = 0.0001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConV1dSmooth(feature_size,hidden_size)
model = model.to(device)
train(model, train_loader, val_loader,test_loader, num_epochs,learning_rate, device)

model.load_state_dict(torch.load(f"./smooth_checkpoints/sc_{args.scenario}_fold_{args.vali_fold}/checkpoint.pth"))

test_dir = f"{folder_path_train}_test"

if args.scenario == 1:
    test_storage = load_data_no_folds(scenario_dir,"test")
else:
    test_storage = load_data_with_folds(scenario_dir,"test")

label_time = pd.Series(range(0,50001,50))

if args.scenario == 1:
     test_file = [fn for fn,_,_ in test_storage]
else:
    test_file = [fn for fn, _,_ in test_storage[f"fold_{args.vali_fold}"]]

true_test = [np.load(f"{test_dir}/{f[:-4]}.npy") for f in test_file]

concat_len = 10
overlap = 10

true_test_feature = [np.array([np.vstack(pred[i-concat_len:i]) for i in range(concat_len,pred.shape[0]+1,overlap)])\
                 for pred in true_test]
true_test_feature= np.vstack(true_test_feature)
true_test_feature = torch.tensor(true_test_feature)

final_inp,final_pred = predict(model,true_test_feature,device)

final_inp_to_save = final_inp.reshape(-1,1000,2)
final_pred_to_save = final_pred.reshape(-1,1000,2)

final_raw_file_pred = final_inp_to_save[:,200:-199,:]
final_smooth_file_pred = final_pred_to_save[:,200:-199,:]

time_stamp = list(range(10000,40001,50))

file_raws = [pd.DataFrame(data = {'time':time_stamp, 'valence':final_raw_file_pred[i,:,0],'arousal':final_raw_file_pred[i,:,1]})\
             for i in range(len(final_raw_file_pred))]
file_smooths = [pd.DataFrame(data = {'time':time_stamp, 'valence':final_smooth_file_pred[i,:,0],'arousal':final_smooth_file_pred[i,:,1]})\
             for i in range(len(final_smooth_file_pred))]

if args.scenario == 1:
    smooth_dir = f"./preds/smooth/scenario_{args.scenario}/test/annotations"
    raw_dir = f"./preds/raw/scenario_{args.scenario}/test/annotations"
else:
    smooth_dir = f"./preds/smooth/scenario_{args.scenario}/fold_{args.vali_fold}/test/annotations"
    raw_dir = f"./preds/raw/scenario_{args.scenario}/fold_{args.vali_fold}/test/annotations"
os.makedirs(smooth_dir,exist_ok = True)
os.makedirs(raw_dir,exist_ok = True)

for f,file in zip(test_file,file_raws):
    file.to_csv(f"{raw_dir}/{f[:-4]}.csv",index = None)

for f,file in zip(test_file,file_smooths):
    file.to_csv(f"{smooth_dir}/{f[:-4]}.csv",index = None)

