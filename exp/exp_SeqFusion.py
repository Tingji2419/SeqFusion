import os
import time
import copy
import warnings
import torch
import torch.nn as nn
import numpy as np
from torch import optim
from exp.exp_basic import Exp_Basic
from transformer_models.models import SeqFusion
from utils.tools import TimeBudget


warnings.filterwarnings('ignore')


class Exp_SeqFusion(Exp_Basic):
    def __init__(self, args):
        super(Exp_SeqFusion, self).__init__(args)
        self.vali_timer = TimeBudget(args.time_budget)
        self.train_timer = TimeBudget(args.time_budget)
        self.test_timer = TimeBudget(args.time_budget)

    def _build_model(self):
        model = SeqFusion.Model(self.args).float()
        if self.args.load_model_path or self.args.load_model_path_list:
            model.load_pretrained(self.args.load_model_path_list)
            print(f'Load All PTM from {self.args.load_model_path_list}')

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model.to(self.device)

    def _select_optimizer(self):
        model_optim = optim.Adam(
            self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        pass

    def train(self, setting):
        print(setting)
        print('Skip training for Exp_SeqFusion')
        return

    def test(self, setting, test=0, selected_model_list=None):

        if self.args.model in ['SeqFusion']:
            setting = 'ALL-ZOO_' + setting    

        if self.args.data in ['m3', 'm4', 'tourism']:
            self.args.pred_len = self.args.test_pred_len
            self.args.seq_padding_size = self.args.seq_len
            self.args.seq_len = self.args.test_seq_len
            test_data, test_loader = self._get_data(flag='test', seq_padding_size=self.args.seq_padding_size)
        else:
            self.args.pred_len = self.args.test_pred_len
            test_data, test_loader = self._get_data(flag='test')

        preds = []
        trues = []

        self.model.eval()
        j = 0
        self.test_timer.start_timer()
        with torch.no_grad():

            for cur_channel in range(self.args.enc_in):
                cur_selected_model_list = selected_model_list[:, :, cur_channel] #only support channel=1
                cur_channel_pred = []
                cur_channel_true = []
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                    
                    batch_x = batch_x[:,:,cur_channel:cur_channel+1]
                    batch_y = batch_y[:,:,cur_channel:cur_channel+1]
                    batch_x_mark = batch_x_mark[:,:,cur_channel:cur_channel+1]
                    batch_y_mark = batch_y_mark[:,:,cur_channel:cur_channel+1]

                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_selected_model_index = cur_selected_model_list[:, i*self.args.batch_size: (i+1)*self.args.batch_size]
                    batch_x = (batch_x, batch_selected_model_index)

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
                    if self.args.label_len == 0:
                        outputs = outputs[:, :self.args.pred_len, :]
                        batch_y = batch_y[:, :self.args.pred_len, :].to(self.device)
                    else:
                        outputs = outputs[:, -self.args.pred_len:, :]
                        batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                    # batch_y = batch_y[:, -self.args.pred_len:,
                    #                   f_dim:].to(self.device)
                    outputs = outputs.detach().cpu().numpy()
                    batch_y = batch_y.detach().cpu().numpy()

                    pred = outputs  # outputs.detach().cpu().numpy()  #[B,T,1]
                    true = batch_y  # batch_y.detach().cpu().numpy()  
                    cur_channel_pred.append(pred)   #[B,T,1] * n
                    cur_channel_true.append(true)

                preds.append(np.concatenate(cur_channel_pred, axis=0))  # [B*n,T,1]* C
                trues.append(np.concatenate(cur_channel_true, axis=0))
        
        self.test_timer.end_timer()
        preds = np.concatenate(preds, axis=-1)  # stack on channels, [B*n,T,C]
        trues = np.concatenate(trues, axis=-1)
        preds = [np.expand_dims(i, 0) for i in preds]  # keep [B,T,C] * n 
        trues = [np.expand_dims(i, 0) for i in trues]
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

        return
