import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from exp.exp_basic import Exp_Basic
import prophet
import pandas as pd

warnings.filterwarnings('ignore')


class Exp_Prophet(Exp_Basic):
    def __init__(self, args):
        super(Exp_Prophet, self).__init__(args)

    def _build_model(self):
        # return prophet.Prophet()
        return []

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        self.train_timer.start_timer()
        for channel in range(train_data.data_y.shape[1]):
            train_df = pd.DataFrame({'y': train_data.data_y.T[channel], 'ds': list(
                pd.to_datetime(train_data.data_stamp_original['date']))})
            self.model.append(prophet.Prophet())
            self.model[-1].fit(train_df)
        self.train_timer.end_timer()
        return

    def test(self, setting, test=0):
        horizon = self.args.pred_len

        test_data, test_loader = self._get_data(flag='test')
        preds, trues = [], []
        self.test_timer.start_timer()
        for channel in range(test_data.data_y.shape[1]):
            test_df = pd.DataFrame({'y': test_data.data_y.T[channel], 'ds': list(
                pd.to_datetime(test_data.data_stamp_original['date']))})
            # predict_frame = self.model[-1].make_future_dataframe(
            #     test_data.data_x.shape[0])
            forecast = self.model[channel].predict(test_df)

            cmp = pd.DataFrame({
                'date': test_df['ds'].values,
                'ds': forecast.ds.values,
                'y': test_df['y'].values,
                'yhat': forecast.yhat.values
            })

            for i in range(self.args.seq_len, cmp.shape[0]-horizon+1):
                pred = cmp[i:i+horizon]['yhat'].values
                true = cmp[i:i+horizon]['y'].values
                pred = pred.reshape(1, -1, 1)
                true = true.reshape(1, -1, 1)
                preds += [pred]
                trues += [true]

        self.test_timer.end_timer()

        return self._save_test_data(setting, preds, trues, self.args.save_result_path)


