import os
# os.chdir("/data/FEDformer/")
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.data_factory import data_provider
from exp.exp_basic_acce import Exp_Basic
from data_provider.data_loader import Dataset_EPiC
from models import FEDformer, Autoformer, Informer, Transformer,FEDformer_EPiC
from utils.tools import EarlyStopping, adjust_learning_rate, visual,load_data_no_folds,load_data_with_folds
from utils.metrics import metric
import argparse
import random
from tqdm import tqdm

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')
# basic config
parser.add_argument('--is_training', type=int, default=1, help='status')
parser.add_argument('--task_id', type=str, default='EPiC', help='task id')
parser.add_argument('--model', type=str, default='FEDformer_EPiC',
                    help='model name, options: [FEDformer, Autoformer, Informer, Transformer]')

# supplementary config for FEDformer model
parser.add_argument('--version', type=str, default='EWDF',
                    help='for FEDformer, there are two versions to choose, options: [Fourier, Wavelets]')
parser.add_argument('--mode_select', type=str, default='random',
                    help='for FEDformer, there are two mode selection method, options: [random, low]')
parser.add_argument('--modes', type=int, default=64, help='modes to be selected random 64')
parser.add_argument('--L', type=int, default=3, help='ignore level')
parser.add_argument('--base', type=str, default='legendre', help='mwt base')
parser.add_argument('--cross_activation', type=str, default='tanh',
                    help='mwt cross atention activation function tanh or softmax')

# data loader
parser.add_argument('--data', type=str, default='EPiC', help='dataset type')
parser.add_argument('--root_path', type=str, default='/data/EPiC_project/EPiC-2023-competition-data-v1.0/data', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, '
                         'S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='m',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '
                         'b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--scenario', type=int, default= 1,
                    help='data scenario, options:[1,2,3,4],')
parser.add_argument('--vali_fold', type=int, default=0,
                    help='which fold to be validation options:[0,1,2,3,4]')
parser.add_argument('--scenario_data_type', type=str, default="train",
                    help='data scenario, options:[train,test],')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=501, help='input sequence length')
parser.add_argument('--label_len', type=int, default=1, help='start token length')
parser.add_argument('--pred_len', type=int, default=10, help='prediction sequence length')
# parser.add_argument('--cross_activation', type=str, default='tanh'

# model define
parser.add_argument('--enc_in', type=int, default=8, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=2, help='decoder input size')
parser.add_argument('--c_out', type=int, default=2, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', default=[24], help='window size of moving average')
parser.add_argument('--factor', type=int, default=3, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='Exp', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type5', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1', help='device ids of multi gpus')

args = parser.parse_args()

setting = '{}_{}_{}_{}_modes{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_id,
                args.model,
                args.mode_select,
                args.version,
                args.modes,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                0)

model = FEDformer_EPiC.Model(args).float()



best_model_path = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
model.load_state_dict(torch.load(best_model_path))


    
scenario_dir = f"{args.root_path}/scenario_{args.scenario}"
print(scenario_dir)

# Loading data
if args.scenario == 1:
    storage = load_data_no_folds(scenario_dir,args.scenario_data_type)
else:
    storage = load_data_with_folds(scenario_dir,args.scenario_data_type)

# Validation Configuration   
if args.scenario == 1:
    num_test = 200
    num_val = 200
    num_val_test = num_val+num_test
else:
    num_validation = len(storage[f"fold_{args.vali_fold}"])//2
    fold_list = list(storage.keys())
    train_fold = [fl for fl in fold_list if fl != f"fold_{args.vali_fold}"]    

overlap = args.label_len-1+ args.pred_len
lab_pred_len = args.label_len-1+ args.pred_len

if args.scenario == 1:
    train_file = [np.array([fn\
                                        for i in range(lab_pred_len,len(label)-num_val_test,overlap)] )\
                                for fn, feature, label in storage]
    valdation_file = [np.array([fn\
                                    for i in range(len(label)-num_val_test,len(label)-num_test,overlap)] )\
                            for fn, feature, label in storage]
    test_file = [np.array([fn\
                                    for i in range(len(label)-num_test,len(label),args.pred_len)] )\
                            for fn, feature, label in storage]
else:
    train_file = [np.array([fn \
                            for i in range(lab_pred_len,len(label),overlap)])\
                    for fl in train_fold for fn, feature, label in storage[fl]]

    valdation_file = [np.array([fn \
                            for i in range(lab_pred_len,len(label),overlap)])\
                    for fn, feature, label in storage[f"fold_{args.vali_fold}"][:num_validation]]

    test_file = [np.array([fn \
                            for i in range(lab_pred_len,len(label),args.pred_len)] )\
                    for fn, feature, label in storage[f"fold_{args.vali_fold}"][num_validation:]]
    
train_file = np.hstack(train_file)
valdation_file = np.hstack(valdation_file)
test_file = np.hstack(test_file)


def data_provider(args, flag):
    Data = Dataset_EPiC
    timeenc = 0 if args.embed != 'timeF' else 1
    shuffle_flag = False
    drop_last = False
    batch_size = 1
    freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        scenario = args.scenario,
        vali_fold = args.vali_fold,
        scenario_data_type = args.scenario_data_type,
        timeenc=timeenc,
        freq=freq
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=drop_last)
    return data_set, data_loader

train_data, train_loader = data_provider(args,'train')
vali_data, vali_loader = data_provider(args,'val')
test_data, test_loader = data_provider(args,'test')

for i, (x,x_mark, t,t_mark) in enumerate(train_loader):
    print(x.shape,x_mark.shape,t.shape,t_mark.shape)
    break
    
device = torch.device('cuda:0')
model = model.to(device)

outputs = []
labels = []
for dl in [train_loader,vali_loader,test_loader]:
    print(len(dl))
    output = []
    label = []
    model.eval()
    with torch.no_grad():
        for i, (batch_x,batch_y,batch_x_mark, batch_y_mark) in tqdm(enumerate(dl)):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
                    # .to(self.device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
            out = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            batch_y = batch_y[:, -args.pred_len:, 0:]
            output.append(out[0].detach().cpu().numpy())
            label.append(batch_y.detach().cpu().numpy())
        outputs.append(output)
        labels.append(label)


train_out = np.vstack([[o] for o in outputs[0]])
vali_out = np.vstack([[o] for o in outputs[1]])
test_out = np.vstack([[o] for o in outputs[-1]])

train_lab = np.vstack(labels[0])
vali_lab = np.vstack(labels[1])
test_lab = np.vstack(labels[-1])
if args.scenario==1:
    folder_path_train = f"./features/scenario_{args.scenario}/fold_0_ll_{args.label_len}"
else:
    folder_path_train = f"./features/scenario_{args.scenario}/fold_{args.vali_fold}_ll_{args.label_len}"

fn_dir = f"{folder_path_train}/file_name" 
out_dir = f"{folder_path_train}/outputs" 
lab_dir = f"{folder_path_train}/labels" 

os.makedirs(fn_dir,exist_ok = True)
os.makedirs(out_dir,exist_ok = True)
os.makedirs(lab_dir,exist_ok = True)

np.save(f"{out_dir}/train_out.npy", train_out)
np.save(f"{out_dir}/vali_out.npy", vali_out)
np.save(f"{out_dir}/test_out.npy", test_out)

np.save(f"{lab_dir}/train_lab.npy", train_lab)
np.save(f"{lab_dir}/vali_lab.npy", vali_lab)
np.save(f"{lab_dir}/test_lab.npy", test_lab)

np.save(f"{fn_dir}/train_file.npy",train_file)
np.save(f"{fn_dir}/valdation_file.npy",valdation_file)
np.save(f"{fn_dir}/test_file.npy",test_file)


# Inference Stage For Testing Set
if args.scenario==1:
    test_storage = load_data_no_folds(scenario_dir,"test")
else:
    test_storage = load_data_with_folds(scenario_dir,"test")

label_time = pd.Series(range(0,50001,50))

if args.scenario==1:
    train_feature = [np.array([feature.loc[feature.time.between(label.time[i-lab_pred_len],label.time[i])].iloc[:,1:].values\
                                    for i in range(lab_pred_len,len(label)-num_val_test,overlap)] )\
                            for _, feature, label in tqdm(storage)]
else:   
    train_feature = [np.array([feature.loc[feature.time.between(label.time[i-lab_pred_len],label.time[i])].iloc[:,1:].values\
                                    for i in range(lab_pred_len,len(label),overlap)] )\
                            for fl in train_fold for fn, feature, label in storage[fl]]
train_feature = np.vstack(train_feature)
scaler = StandardScaler()
scaler.fit(train_feature.reshape(-1, train_feature.shape[-1]))

# Prepare Features
if args.scenario==1:

    test_feature = [np.array([feature.loc[feature.time.between(label_time[i-lab_pred_len],label_time[i])].iloc[:,1:].values\
                for i in range(lab_pred_len,len(label_time),overlap)] )\
        for _, feature, label in tqdm(test_storage)]
    test_fea_timestamp = [np.array([feature.loc[feature.time.between(label_time[i-lab_pred_len],label_time[i])].iloc[:,:1].values\
                            for i in range(lab_pred_len,len(label_time),overlap)] )\
                    for _, feature, label in test_storage]
    test_lab_timestamp = [np.array([label_time[i-lab_pred_len:i+1].values\
                            for i in range(lab_pred_len,len(label_time),overlap)] )\
                    for _, feature, label in test_storage]
    test_file = [np.array([fn\
                                        for i in range(lab_pred_len,len(label_time),overlap)])\
                                for fn, feature, label in test_storage]
else:
    test_feature = [np.array([feature.loc[feature.time.between(label_time[i-lab_pred_len],label_time[i])].iloc[:,1:].values\
                    for i in range(lab_pred_len,len(label_time),overlap)] )\
            for _, feature, label in tqdm(test_storage[f"fold_{args.vali_fold}"])]
    test_fea_timestamp = [np.array([feature.loc[feature.time.between(label_time[i-lab_pred_len],label_time[i])].iloc[:,:1].values\
                                for i in range(lab_pred_len,len(label_time),overlap)] )\
                        for _, feature, label in test_storage[f"fold_{args.vali_fold}"]]

    test_lab_timestamp = [np.array([label_time[i-lab_pred_len:i+1].values\
                                for i in range(lab_pred_len,len(label_time),overlap)] )\
                        for _, feature, label in test_storage[f"fold_{args.vali_fold}"]]

    test_file = [np.array([fn\
                                            for i in range(lab_pred_len,len(label_time),overlap)] )\
                                    for fn, feature, label in test_storage[f"fold_{args.vali_fold}"]]




def prediction(batch_x,batch_y,batch_x_mark,batch_y_mark,device):
    model.eval()
    with torch.no_grad(): 
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)
        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
                # .to(self.device)
        dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
        out = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        # batch_y = batch_y[:, -args.pred_len:, 0:]
        pred = out[0].detach().cpu().numpy()
    return pred

# Prepare Initial Value for Inference
if args.scenario==1:
    # For scenario 1 
    # Use the last label from the training file as initial value
    train_test_lab = [np.array([label.iloc[i-lab_pred_len:i+1].iloc[:,1:].values\
                            for i in range(len(label)-num_test,len(label),args.pred_len)])\
                    for _, _, label in storage]
    batch_y_t = [te[-1,-1] for te in train_test_lab] 
    
else:
    # For the rest scenarios
    # Use the neutral (5.0,5.0) as initial value
    batch_y_t = [np.array([5.0,5.0]) for _ in test_lab_timestamp] 
    
preds = []
inputs_y = []

# Inference Loop
for te_fea_f, te_fea_time_f, ini_b_y_f, te_lab_time_f in tqdm(zip(test_feature,test_fea_timestamp,batch_y_t,test_lab_timestamp)):
    f_pred = []
    count = 0
    te_fea_f = scaler.transform(te_fea_f.reshape(-1, te_fea_f.shape[-1])).reshape(te_fea_f.shape)
    for te_fea, te_fea_time, te_lab_time in zip(te_fea_f,te_fea_time_f,te_lab_time_f):
        te_lab_time = te_lab_time.reshape(te_lab_time.shape[0],1)
        te_fea = torch.tensor(te_fea).view(1,te_fea.shape[0],te_fea.shape[1]).to(device)
        te_fea_time = torch.tensor(te_fea_time).view(1,te_fea_time.shape[0],te_fea_time.shape[1]).to(device)
        te_lab_time = torch.tensor(te_lab_time).view(1,te_lab_time.shape[0],te_lab_time.shape[1]).to(device)
        b_y = torch.tensor(ini_b_y_f).view(1,2).to(device)
        zeros = torch.zeros([args.pred_len, 2],device=device)
        if count == 0:
            # For the first frame use the prepared initial value for pesudo label
            # Post Padding Zeros
            pesudo_y = torch.cat([b_y, zeros], dim=0)
        else:
            # The rest frames will use the previous prediction
            p_y = f_pred[-1][-1]
            inputs_y.append(p_y)
            p_y = torch.tensor(p_y).view(1,2).to(device)
            # Post padding zeros
            pesudo_y = torch.cat([p_y, zeros], dim=0)
        pesudo_y = pesudo_y.view(1,pesudo_y.shape[0],pesudo_y.shape[1])
        pred = prediction(te_fea,pesudo_y,te_fea_time,te_lab_time,device)
        f_pred.append(pred)
        count+=1
    preds.append(f_pred)
    
if args.scenario==1:
    fns = [fn for fn,_,_ in test_storage]
else:
    fns = [fn for fn,_,_ in test_storage[f"fold_{args.vali_fold}"]]

folder_path_test = f"{folder_path_train}_test"
os.makedirs(folder_path_test,exist_ok = True)

for pred, fn in zip(preds,fns):
    np.save(f"{folder_path_test}/{fn[:-4]}.npy", pred)
