import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from data_provider.m4 import M4Meta
from exp.exp_basic import Exp_Basic
from torch import optim
from transformer_models.models import Nbeats
from utils.losses import smape_loss
from utils.metrics import metric
from utils.metrics_with_nan import metric_with_nan
from utils.tools import EarlyStopping, TimeBudget, adjust_learning_rate
from transformer_models.models import FEDformer, Autoformer, Informer, Transformer, DLinear
warnings.filterwarnings('ignore')


class Exp_Nbeats(Exp_Basic):
    def __init__(self, args):
        super(Exp_Nbeats, self).__init__(args)
        self.vali_timer = TimeBudget(args.time_budget)
        self.train_timer = TimeBudget(args.time_budget)
        self.test_timer = TimeBudget(args.time_budget)

    def _build_model(self):
        # self.args.frequency_map = M4Meta.frequency_map[self.args.seasonal_patterns]
        # model = Nbeats.Model(self.args).float()
        model_dict = {
            'FEDformer': FEDformer,
            'FEDformer-w': FEDformer,
            'FEDformer-f': FEDformer,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'nbeats': Nbeats,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model.to(self.device)

    def _select_optimizer(self):
        model_optim = optim.Adam(
            self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self, loss_name='SMAPE'):
        return smape_loss()

    def vali(self, train_loader, vali_loader, criterion):
        x, _ = train_loader.dataset.last_insample_window()
        y = vali_loader.dataset.timeseries
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.unsqueeze(-1)

        self.model.eval()
        with torch.no_grad():
            # decoder input
            B, _, C = x.shape
            dec_inp = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            dec_inp = torch.cat([x[:, -self.args.label_len:, :], dec_inp], dim=1).float()
            # encoder - decoder
            outputs = torch.zeros((B, self.args.pred_len, C)).float()  # .to(self.device)
            id_list = np.arange(0, B, 500)  # validation set size
            id_list = np.append(id_list, B)
            for i in range(len(id_list) - 1):
                outputs[id_list[i]:id_list[i + 1], :, :] = self.model(x[id_list[i]:id_list[i + 1]], None,
                                                                      dec_inp[id_list[i]:id_list[i + 1]],
                                                                      None).detach().cpu()
            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            pred = outputs
            true = torch.from_numpy(np.array(y))
            true = true[:, :self.args.pred_len]
            true = true
            batch_y_mark = torch.ones(true.shape)
            loss = criterion(pred[:, :, 0], true, batch_y_mark)

        self.model.train()
        return loss

    def train(self, setting):
        print(setting)
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')


        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=False)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion(self.args.loss)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                outputs = self.model(batch_x, None, dec_inp, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                batch_y_mark = batch_y_mark[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss_value = criterion(outputs, batch_y, batch_y_mark)
                loss = loss_value  # + loss_sharpness * 1e-5
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                # print('before backward')
                loss.backward()
                # print('get loss backward')
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(train_loader, vali_loader, criterion)
            test_loss = vali_loss
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        if self.args.source_data is not None:
            from utils.arg_resolver import setting_string
            import copy
            args = copy.deepcopy(self.args)
            args.data = args.source_data
            model_path = setting_string(args, 0)
        else:
            model_path = setting
        print('load model from:', model_path)
        best_model_path = os.path.join('./checkpoints/',
                                       model_path) + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        if self.args.data in ['m3', 'm4', 'tourism']:
            train_data, train_loader = self._get_data(flag='train')
            vali_data, vali_loader = self._get_data(flag='val')
            criterion = self._select_criterion(self.args.loss)
            return self._save_test_data(setting, None, None, self.args.save_result_path,
                                        loss=self.vali(train_loader, vali_loader, criterion))
        else:
            test_data, test_loader = self._get_data(flag='test')
            preds = []
            trues = []

            self.model.eval()
            self.test_timer.start_timer()
            with torch.no_grad():
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                    # print(batch_x.shape, batch_y.shape, batch_x_mark.shape, batch_y_mark.shape)
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)

                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_inp = torch.zeros_like(
                        batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat(
                        [batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model(
                                    batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(
                                    batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0

                    batch_y = batch_y[:, -self.args.pred_len:,
                              f_dim:].to(self.device)
                    outputs = outputs.detach().cpu().numpy()
                    batch_y = batch_y.detach().cpu().numpy()

                    pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                    true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()
                    # print(pred.shape, true.shape)
                    preds.append(pred)
                    trues.append(true)
            self.test_timer.end_timer()

            return self._save_test_data(setting, preds, trues, self.args.save_result_path)

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
                dec_inp = torch.zeros_like(
                    batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        # folder_path = './results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        # np.save(folder_path + 'real_prediction.npy', preds)

        return

    def reset(self):
        self.model = self._build_model()
        return

    def _save_test_data(self, setting, preds, trues, save_result_path='result_tmp.txt', loss=None):
        if loss is not None:
            print("smape:{}".format(loss))
            f = open(f"{save_result_path}", 'a')
            f.write(setting + "  \n")
            f.write('smape:{}'.format(loss))
            f.write('\n')
            f.write('\n')
            f.close()
            return loss
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        # print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        if preds.shape[-1] != trues.shape[-1]:
            preds = preds.transpose(0, 2, 1)
        print('test shape:', preds.shape, trues.shape)

        # result save
        # folder_path = './results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)
        mae, mse, rmse, mape, smape, mspe, nd = metric(preds, trues)
        nan_count = np.isnan(preds).sum()
        if np.isnan(mse):
            mae, mse, rmse, mape, smape, mspe, nd = metric_with_nan(preds, trues)

        print('mae:{}, mse:{}, rmse:{}, mape:{}, smape:{}, mspe:{}, nd:{}, nan:{}'.format(mae, mse, rmse, mape, smape,
                                                                                          mspe, nd, nan_count))
        f = open(f"{save_result_path}", 'a')
        f.write(setting + "  \n")
        f.write('mae:{}, mse:{}, rmse:{}, mape:{}, smape:{}, mspe:{}, nd:{}, nan:{}'.format(mae, mse, rmse, mape, smape,
                                                                                            mspe, nd, nan_count))
        f.write('\n')
        f.write('\n')
        f.close()

        output = {
            'metrics': {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'mape': mape,
                'mspe': mspe,
            },
            'train_timer': self.train_timer.total_time,
            'vali_timer': self.vali_timer.total_time,
            'test_timer': self.test_timer.total_time,
            'args': self.args
        }
        print(output)

        # np.save(folder_path + 'metrics.npy', output)
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        return mae, mse, rmse, mape, mspe
