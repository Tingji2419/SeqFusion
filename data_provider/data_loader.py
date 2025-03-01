import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from utils.timefeatures import time_features
from sklearn.preprocessing import StandardScaler
from data_provider.m4 import M4Dataset, M4Meta, M3Meta
import warnings

warnings.filterwarnings('ignore')

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', 
                 scaler=StandardScaler(), train_budget=None, **kwargs):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.scaler = scaler
        self.train_budget = train_budget

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        # print(num_train, num_vali, num_test, len(df_raw))
        train_start = 0
        if self.train_budget:
            train_start = max(train_start, num_train -
                              self.seq_len - self.train_budget)

        border1s = [train_start, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[0:border2s[0]]# # TODO: the original ForecastPFN code use df_data[0:border2s[0]], this is wrong when budget=x, should be df_data[border1s[0]: border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
            # for gpt
            cols_to_scale = df_raw.columns[1:]
            # print(cols_to_scale)
            self.data_stamp_original = df_raw[border1:border2]
            scaled_features = self.scaler.transform(self.data_stamp_original[cols_to_scale].values)
            self.data_stamp_original[cols_to_scale] = scaled_features
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        # cols_to_scale = df_raw.columns[1:]

#         self.data_stamp_original = df_raw[border1:border2]
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        # print(border1, border2, len(self.data_x), len(self.data_y), len(self.data_stamp))

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        seq_x_original = self.data_stamp_original['date'].values[s_begin:s_end]
        seq_y_original = self.data_stamp_original['date'].values[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark#, seq_x_original, seq_y_original

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None,
                 scaler=StandardScaler(), **kwargs):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.scaler = scaler
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_M4(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Hourly', **kwargs):
        # size [seq_len, label_len, pred_len]
        # init
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        # self.pred_len = M4Meta.horizons_map[seasonal_patterns]
        # self.seq_len = 2 * self.pred_len
        # self.label_len = self.pred_len
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        # self.window_sampling_limit = self.seq_len
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        path_train = os.path.join(self.root_path,
                                  f'{self.seasonal_patterns}-sl{self.seq_len}-pl{self.pred_len}-train.csv')
        path_test = os.path.join(self.root_path,
                                 f'{self.seasonal_patterns}-sl{self.seq_len}-pl{self.pred_len}-test.csv')
        if not os.path.exists(path_train) or not os.path.exists(path_test):
            # Load both training and test datasets
            train_dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
            test_dataset = M4Dataset.load(training=False, dataset_file=self.root_path)

            # Filter the datasets by seasonal patterns
            train_values = train_dataset.values[train_dataset.groups == self.seasonal_patterns]
            test_values = test_dataset.values[test_dataset.groups == self.seasonal_patterns]

            # Concatenate the training and test values for each corresponding series
            self.timeseries = [np.concatenate([train_series[~np.isnan(train_series)],
                                               test_series[~np.isnan(test_series)]])
                               for train_series, test_series in zip(train_values, test_values)]
            self.ids = train_dataset.ids[train_dataset.groups == self.seasonal_patterns]
            data_train = []
            data_test = []
            max_train = 0
            max_test = 0
            for i in range(len(self.timeseries)):
                data_train.append([self.ids[i]] + list(self.timeseries[i][:-self.pred_len]))
                max_train = max(max_train, len(data_train[-1]))
                data_test.append([self.ids[i]] + list(self.timeseries[i][-self.pred_len:]))
                max_test = max(max_test, len(data_test[-1]))
            columns_train = ['V' + str(i+1) for i in range(max_train)]
            columns_test = ['V' + str(i+1) for i in range(max_test)]
            data_train = pd.DataFrame(data_train, columns=columns_train)
            data_test = pd.DataFrame(data_test, columns=columns_test)
            path_train = os.path.join(self.root_path, f'{self.seasonal_patterns}-sl{self.seq_len}-pl{self.pred_len}-train.csv')
            path_test = os.path.join(self.root_path, f'{self.seasonal_patterns}-sl{self.seq_len}-pl{self.pred_len}-test.csv')
            data_train.to_csv(path_train, index=False)
            data_test.to_csv(path_test, index=False)
        if self.flag == 'train':
            dataset = pd.read_csv(path_train)
        else:
            dataset = pd.read_csv(path_test)
        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[:, 1:].astype(float)])
        self.timeseries = [ts for ts in training_values]
        self.ids = dataset.values[:, 0]

        # if self.flag == 'train':
        #     dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        # else:
        #     dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        # training_values = np.array(
        #     [v[~np.isnan(v)] for v in
        #      dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
        # self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        # self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           cut_point - self.label_len:min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask

class Dataset_M3(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Hourly', data='m3', **kwargs):
        # size [seq_len, label_len, pred_len]
        # init
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path
        self.data = data
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M3Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        path_train = os.path.join(self.root_path,
                                  f'{self.seasonal_patterns}-sl{self.seq_len}-pl{self.pred_len}-train.csv')
        path_test = os.path.join(self.root_path,
                                 f'{self.seasonal_patterns}-sl{self.seq_len}-pl{self.pred_len}-test.csv')
        if not os.path.exists(path_train) or not os.path.exists(path_test):
            from sktime.datasets import load_tsf_to_dataframe
            filepath = os.path.join(self.root_path, f'{self.data}_{self.seasonal_patterns.lower()}_dataset.tsf')
            df, metadata = load_tsf_to_dataframe(filepath)
            self.ids = []
            self.timeseries = []
            for id, timestamp in list(df.index):
                if id not in self.ids:
                    self.ids.append(id)
                    self.timeseries.append([])
                self.timeseries[-1].append(df.loc[(id, timestamp)].values[0])
            data_train = []
            data_test = []
            max_train = 0
            max_test = 0
            for i in range(len(self.timeseries)):
                data_train.append([self.ids[i]] + list(self.timeseries[i][:-self.pred_len]))
                max_train = max(max_train, len(data_train[-1]))
                data_test.append([self.ids[i]] + list(self.timeseries[i][-self.pred_len:]))
                max_test = max(max_test, len(data_test[-1]))
            columns_train = ['V' + str(i+1) for i in range(max_train)]
            columns_test = ['V' + str(i+1) for i in range(max_test)]
            data_train = pd.DataFrame(data_train, columns=columns_train)
            data_test = pd.DataFrame(data_test, columns=columns_test)
            path_train = os.path.join(self.root_path, f'{self.seasonal_patterns}-sl{self.seq_len}-pl{self.pred_len}-train.csv')
            path_test = os.path.join(self.root_path, f'{self.seasonal_patterns}-sl{self.seq_len}-pl{self.pred_len}-test.csv')
            data_train.to_csv(path_train, index=False)
            data_test.to_csv(path_test, index=False)
        if self.flag == 'train':
            dataset = pd.read_csv(path_train)
        else:
            dataset = pd.read_csv(path_test)
        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[:, 1:].astype(float)])
        self.timeseries = [ts for ts in training_values]
        self.ids = dataset.values[:, 0]

        # if self.flag == 'train':
        #     dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        # else:
        #     dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        # training_values = np.array(
        #     [v[~np.isnan(v)] for v in
        #      dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
        # self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        # self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           cut_point - self.label_len:min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask


class Dataset_TSF(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path=None,
                 target='OT', scale=False, timeenc=0, freq='Daily',
                 seasonal_patterns=None,
                 percent=10, max_len=-1, train_all=False,
                 seq_padding_size=None, padding_type='average', **kwargs):

        self.train_all = train_all
        self.scale = scale
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.seq_padding_size = seq_padding_size
        self.padding_type = padding_type
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.percent = percent
        self.max_len = max_len
        if self.max_len == -1:
            self.max_len = 1e8

        self.root_path = root_path
        self.data_path = data_path
        self.timeseries = self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        from utils.tools import convert_tsf_to_dataframe
        print('Loading data from', os.path.join(self.root_path, self.data_path))
        df, frequency, forecast_horizon, contain_missing_values, contain_equal_length = convert_tsf_to_dataframe(
            os.path.join(self.root_path,
                         self.data_path))
        self.freq = frequency
        # print(df.head())
        def dropna(x):
            return x[~np.isnan(x)]

        timeseries = [dropna(ts).astype(np.float32) for ts in df.series_value]

        if self.scale:
            all_train_data = np.concatenate([ts[:len(timeseries) - self.pred_len] for ts in timeseries])
            self.scaler.fit(all_train_data.reshape(-1, 1))
            timeseries = [self.scaler.transform(ts.reshape(-1, 1)).flatten() for ts in timeseries]

        self.tot_len = 0
        self.len_seq = []
        self.seq_id = []
        for i in range(len(timeseries)):
            res_len = max(self.pred_len + self.seq_len - timeseries[i].shape[0], 0)
            pad_zeros = np.zeros(res_len)
            timeseries[i] = np.hstack([pad_zeros, timeseries[i]])

            _len = timeseries[i].shape[0]
            train_len = _len - self.pred_len
            if self.train_all:
                border1s = [0, 0, train_len - self.seq_len]
                border2s = [train_len, train_len, _len]
            else:
                border1s = [0, train_len - self.seq_len - self.pred_len, train_len - self.seq_len]
                border2s = [train_len - self.pred_len, train_len, _len]
            border2s[0] = (border2s[0] - self.seq_len) * self.percent // 100 + self.seq_len
            # print("_len = {}".format(_len))
            # print('seq_len = {}, pred_len = {}'.format(self.seq_len, self.pred_len))
            # print("border1s = {}, border2s = {}".format(border1s, border2s))
            curr_len = border2s[self.set_type] - max(border1s[self.set_type], 0) - self.pred_len - self.seq_len + 1
            # print("curr_len = {}".format(curr_len))
            curr_len = max(0, curr_len)

            self.len_seq.append(np.zeros(curr_len) + self.tot_len)
            self.seq_id.append(np.zeros(curr_len) + i)
            self.tot_len += curr_len

        self.len_seq = np.hstack(self.len_seq)
        self.seq_id = np.hstack(self.seq_id)

        return timeseries

    def __getitem__(self, index):
        len_seq = self.len_seq[index]
        seq_id = int(self.seq_id[index])
        index = index - int(len_seq)

        _len = self.timeseries[seq_id].shape[0]
        train_len = _len - self.pred_len
        if self.train_all:
            border1s = [0, 0, train_len - self.seq_len]
            border2s = [train_len, train_len, _len]
        else:
            border1s = [0, train_len - self.seq_len - self.pred_len, train_len - self.seq_len]
            border2s = [train_len - self.pred_len, train_len, _len]
        border2s[0] = (border2s[0] - self.seq_len) * self.percent // 100 + self.seq_len

        s_begin = index + border1s[self.set_type]
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.pred_len + self.label_len
        if self.set_type == 2:
            s_end = -self.pred_len

        data_x = self.timeseries[seq_id][s_begin:s_end]
        data_y = self.timeseries[seq_id][r_begin:r_end]

        # padding
        if self.seq_padding_size:
            padding_needed = self.seq_padding_size - len(data_x)
            if padding_needed > 0:
                if self.padding_type == 'average':
                    # Compute average padding
                    padding_value = np.mean(data_x)
                    padding_array = np.full(padding_needed, padding_value)
                elif self.padding_type == 'zero':
                    # Zero padding
                    padding_array = np.zeros(padding_needed)
                # Apply padding to data_x
                data_x = np.concatenate((padding_array, data_x))
            else:
                data_x = data_x[-self.seq_padding_size:]

        data_x = np.expand_dims(data_x, axis=-1)
        data_y = np.expand_dims(data_y, axis=-1)
        # if self.set_type == 2:
        #     print("data_x.shape = {}, data_y.shape = {}".format(data_x.shape, data_y.shape))

        return data_x, data_y, data_x, data_y

    def __len__(self):
        if self.set_type == 0:
            # return self.tot_len
            return min(self.max_len, self.tot_len)
        else:
            return self.tot_len

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
