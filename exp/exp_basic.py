import os
import torch
import numpy as np
from data_provider.data_factory import data_provider
from utils.tools import TimeBudget
from utils.metrics import metric
from utils.metrics_with_nan import metric_with_nan

class Exp_Basic(object):
    def __init__(self, args):
        if args is not None:
            self.args = args
            self.device = self._acquire_device()
            self.model = self._build_model()
            self.vali_timer = TimeBudget(args.time_budget)
            self.train_timer = TimeBudget(args.time_budget)
            self.test_timer = TimeBudget(args.time_budget)

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self, flag, seq_padding_size=None):
        data_set, data_loader = data_provider(self.args, flag, seq_padding_size=seq_padding_size)
        return data_set, data_loader

    def _save_test_data(self, setting, preds, trues, save_result_path='result_tmp.txt'):
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

        # print('mae:{}, mse:{}, rmse:{}, mape:{}, smape:{}, mspe:{}, nd:{}, nan:{}'.format(mae, mse, rmse, mape, smape, mspe, nd, nan_count))
        f = open(f"{save_result_path}", 'a')
        f.write(setting + "  \n")
        f.write('mae:{}, mse:{}, rmse:{}, mape:{}, smape:{}, mspe:{}, nd:{}, nan:{}'.format(mae, mse, rmse, mape, smape, mspe, nd, nan_count))
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

        }
        print(output)
        print()


        return mae, mse, rmse, mape, mspe

    def vali(self, *args, **kwargs):
        pass

    def train(self, *args, **kwargs):
        pass

    def test(self, *args, **kwargs):
        pass

    def reset(self, *args, **kwargs):
        pass
