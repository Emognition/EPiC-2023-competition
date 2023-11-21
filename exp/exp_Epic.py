import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import tqdm
from data_provider.data_factory import data_provider
from exp.exp_basic_acce import Exp_Basic
from models import FEDformer, Autoformer, Informer, Transformer,FEDformer_EPiC
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric



warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'FEDformer_EPiC': FEDformer_EPiC,
            'FEDformer': FEDformer,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        # if self.args.use_multi_gpu and self.args.use_gpu:
        #     model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # criterion = nn.MSELoss()
        criterion = nn.SmoothL1Loss(beta=2.0)
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        total_mse =  []

        mse_loss = nn.MSELoss()
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float()
                # .to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float()
                # .to(self.device)
                batch_y_mark = batch_y_mark.float()
                # .to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                # .to(self.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                # batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                # .contiguous()
                self.accelerator.wait_for_everyone()
                all_pred,all_targets =  self.accelerator.gather_for_metrics((outputs.contiguous(), batch_y.contiguous()))
                # self.accelerator.print(all_pred.shape,all_targets.shape)
                if self.accelerator.is_local_main_process:
                    # if self.accelerator.is_local_main_process:
                    # pred = all_pred.detach().cpu().numpy()
                    # true = all_targets.detach().cpu().numpy()
                    all_pred = all_pred.detach().cpu()
                    all_targets = all_targets.detach().cpu()
                    all_targets = all_targets[:,-self.args.pred_len:,f_dim:]

                    loss = criterion(all_pred, all_targets)
                    mse_l = mse_loss(all_pred, all_targets)

                    total_loss.append(loss)
                    total_mse.append(mse_l)
        # self.accelerator.print(len(total_loss))
        if self.accelerator.is_local_main_process:
            total_loss = np.average(total_loss)
            total_mse  = np.average(total_mse)
        self.accelerator.wait_for_everyone()
        self.model.train()
        return total_loss,total_mse

    def train(self, setting):
        with self.accelerator.local_main_process_first():
            train_data, train_loader = self._get_data(flag='train')
            vali_data, vali_loader = self._get_data(flag='val')
            test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience,verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        



        self.model,model_optim, train_loader,vali_loader,test_loader = self.accelerator.prepare(self.model,model_optim,train_loader,vali_loader,test_loader)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                with self.accelerator.accumulate(self.model):
                    model_optim.zero_grad()
                    batch_x = batch_x.float()
                    # .to(self.device)

                    batch_y = batch_y.float()
                    # .to(self.device)
                    batch_x_mark = batch_x_mark.float()
                    # .to(self.device)
                    batch_y_mark = batch_y_mark.float()
                    # .to(self.device)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    # .to(self.device)
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())
                    self.accelerator.backward(loss)
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                    model_optim.step()

            # if self.accelerator.sync_gradients:
            #     self.accelerator.log({"loss_train":self.accelerate.gather(avg_loss / self.accelerator.gradient_accumulation_steps)})
                # avg_loss = 0

                if (i + 1) % 100 == 0 and self.accelerator.is_local_main_process:
                    self.accelerator.print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    self.accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            # self.accelerator.wait_for_everyone()
            if self.accelerator.is_local_main_process:
                self.accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
                train_loss = np.average(train_loss)
            self.accelerator.wait_for_everyone()
            vali_loss, vali_mse = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_mse = self.vali(test_data, test_loader, criterion)
            if self.accelerator.is_local_main_process:
                self.accelerator.print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f} | Vali MSE: {5:.7f} Test MSE: {6:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss,vali_mse,test_mse))
                    # self.accelerator.wai
            self.accelerator.wait_for_everyone()
            state = self.accelerator.get_state_dict(self.model)
                # unwrapped_model  = self.accelerator.unwrap_model(self.model)
            if self.accelerator.is_local_main_process:
                early_stopping(vali_loss, state, path)
            early_stop_tensor = torch.tensor([early_stopping.early_stop], dtype=torch.bool).to(self.device)
            early_stop_list = self.accelerator.gather(early_stop_tensor)
            early_stop = torch.any(early_stop_list).item()
            if early_stop:
                if self.accelerator.is_local_main_process:
                    self.accelerator.print("Early stopping")
                break
            with self.accelerator.local_main_process_first():
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        self.accelerator.wait_for_everyone()
        with self.accelerator.local_main_process_first():
            best_model_path = path + '/' + 'checkpoint.pth'
            unwrapped_model  = self.accelerator.unwrap_model(self.model)
            unwrapped_model.load_state_dict(torch.load(best_model_path))
            # self.model = unwrapped_model

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        # self.accelerator.wait_for_everyone()
        # unwrapped_model = self.accelerator.unwrap_model(self.model)

        if test:
            with self.accelerator.local_main_process_first():
                print('loading model')
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                unwrapped_model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
                # self.accelerator.prepare(self.model,test_loader)

            # self.model = unwrapped_model
        # if self.accelerator.is_local_main_process:
        test_loader = self.accelerator.prepare(test_loader)
        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.accelerator.wait_for_everyone()
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float()
                # .to(self.device)
                batch_y = batch_y.float()
                # .to(self.device)

                batch_x_mark = batch_x_mark.float()
                # .to(self.device)
                batch_y_mark = batch_y_mark.float()
                # .to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                # .float()
                # .to(self.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                self.accelerator.wait_for_everyone()
                # all_pred = self.accelerator.gather(outputs.contiguous()).detach().cpu()
                # all_targets = self.accelerator.gather(batch_y.contiguous()).detach().cpu()
                all_pred,all_targets =  self.accelerator.gather_for_metrics((outputs.contiguous(), batch_y.contiguous()))
                if self.accelerator.is_local_main_process:
                    # if self.accelerator.is_local_main_process:
                    # pred = all_pred.detach().cpu().numpy()
                    # true = all_targets.detach().cpu().numpy()
                    # all_targets = all_targets[:,-self.args.pred_len:,f_dim:]
                    all_targets = all_targets[:, -self.args.pred_len:, f_dim:]
                    # .to(self.device)
                    all_pred = all_pred.detach().cpu().numpy()
                    all_targets = all_targets.detach().cpu().numpy()
                    # pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                    # true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()
                    preds.append(all_pred)
                    trues.append(all_targets)
                    if i % 20 == 0:
                        input = batch_x.detach().cpu().numpy()
                        # gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                        # pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                        gt_v =  all_targets[0, :, 0]
                        pd_v =  all_pred[0, :, 0]
                        visual(gt_v, pd_v, os.path.join(folder_path, str(i) + '_v.pdf'))
                        gt_a =  all_targets[0, :, -1]
                        pd_a =  all_pred[0, :, -1]
                        visual(gt_a, pd_a, os.path.join(folder_path, str(i) + '_a.pdf'))

        if self.accelerator.is_local_main_process:
            preds = np.vstack(preds)
            trues = np.vstack(trues)
            print('test shape:', preds.shape, trues.shape)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            print('test shape:', preds.shape, trues.shape)

            # result save
            folder_path = './results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # preds = test_data.inverse_transform(preds)
            # trues = test_data.inverse_transform(trues)
            mae, mse, rmse, mape, mspe = metric(preds, trues)
            print('mse:{}, mae:{}'.format(mse, mae))
            f = open("result.txt", 'a')
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}'.format(mse, mae))
            f.write('\n')
            f.write('\n')
            f.close()

            np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
            np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
