import pmdarima
import warnings
import pandas as pd
from exp.exp_basic import Exp_Basic

warnings.filterwarnings('ignore')


class Exp_Arima(Exp_Basic):
    def __init__(self, args):
        super(Exp_Arima, self).__init__(args)

    def _build_model(self):
        return []


    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        self.train_timer.start_timer()
        for channel in range(train_data.data_y.shape[1]):
            train_df = pd.DataFrame({'y': train_data.data_y.T[channel], 'ds': list(
                pd.to_datetime(train_data.data_stamp_original['date']))})
            self.model.append(pmdarima.auto_arima(train_df.y.values))
        self.train_timer.end_timer()
        return

    def test(self, setting, test=0):
        horizon = self.args.pred_len

        test_data, test_loader = self._get_data(flag='test')
        self.test_timer.start_timer()
        preds, trues = [], []
        for channel in range(test_data.data_y.shape[1]):
            test_df = pd.DataFrame({'y': test_data.data_y.T[channel], 'ds': list(
                pd.to_datetime(test_data.data_stamp_original['date']))})

            cmp = pd.DataFrame({
                'date': test_df['ds'].values,
                'y': test_df['y'].values,
                'yhat': self.model[channel].predict(test_df.shape[0])
            })

        # preds, trues = [], []

            for i in range(self.args.seq_len, cmp.shape[0]-horizon+1):
               pred = cmp[i:i+horizon]['yhat'].values
               true = cmp[i:i+horizon]['y'].values
               pred = pred.reshape(1, -1, 1)
               true = true.reshape(1, -1, 1)
               preds += [pred]
               trues += [true]

        self.test_timer.end_timer()
        
        return self._save_test_data(setting, preds, trues, self.args.save_result_path)
