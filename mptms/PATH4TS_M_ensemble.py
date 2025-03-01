import math
import pickle
import numpy as np
from scipy.spatial.distance import cdist 
from SimMTM.models.SimMTM_zoo import Model as SimMTM_zoo


class Config:
    def __init__(self, **kwargs):
        # Set default configuration values
        self.defaults = {
            'task_name': 'pretrain',
            'is_training': 1,
            'model_id': 'SimMTM',
            'model': 'SimMTM',
            'data': '',
            'root_path': './datasets',
            'data_path': 'ETTh1.csv',
            'features': 'M',
            'target': 'OT',
            'freq': 'h',
            'checkpoints': './outputs/checkpoints/',
            'pretrain_checkpoints': './SimMTM/pretrain_checkpoints/',
            'transfer_checkpoints': 'ckpt_best.pth',
            'load_checkpoints': None,
            'select_channels': 1.0,
            'seq_len': 36,
            'label_len': 0,
            'pred_len': 12,
            'seasonal_patterns': 'Monthly',
            'top_k': 5,
            'num_kernels': 3,
            'enc_in': 1,
            'dec_in': 1,
            'c_out': 1,
            'd_model': 128,
            'n_heads': 16,
            'e_layers': 2,
            'd_layers': 1,
            'd_ff': 64,
            'moving_avg': 25,
            'factor': 1,
            'distil': True,
            'dropout': 0.1,
            'fc_dropout': 0.0,
            'head_dropout': 0.1,
            'embed': 'timeF',
            'activation': 'gelu',
            'output_attention': False,
            'individual': 0,
            'pct_start': 0.3,
            'patch_len': 12,
            'stride': 12,
            'num_workers': 5,
            'itr': 1,
            'train_epochs': 10,
            'batch_size': 32,
            'patience': 3,
            'learning_rate': 0.0001,
            'des': 'test',
            'loss': 'MSE',
            'lradj': 'type1',
            'use_amp': False,
            'use_gpu': True,
            'gpu': 0,
            'use_multi_gpu': False,
            'devices': '0',
            'lm': 3,
            'positive_nums': 2,
            'rbtp': 1,
            'temperature': 0.02,
            'masked_rule': 'geometric',
            'mask_rate': 0.5
        }
        
        # Override default values with any user-provided values
        for key, value in self.defaults.items():
            setattr(self, key, kwargs.get(key, value))

    def update(self, **kwargs):
        """Update configuration settings."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise KeyError(f"Invalid configuration key: {key}")

    def __str__(self):
        return '\n'.join(f"{key}: {getattr(self, key)}" for key in sorted(self.defaults.keys()))




def adjust_samples(samples, desired_length=36, find_strategy='mean'):
    """
    Adjusts the samples to have a specific number of timestamps by padding or truncating.
    :return: Adjusted samples of shape [num_samples, desired_length, dim_features].
    """
    num_samples, current_length, dim_features = samples.shape

    if current_length < desired_length:
        # Calculate the amount of padding needed
        padding_length = desired_length - current_length
        # Create the padding array
        padding = np.zeros((num_samples, padding_length, dim_features))
        # Concatenate the padding ahead of the existing samples
        adjusted_samples = np.concatenate((padding, samples), axis=1)
    else:
        if find_strategy == 'mean':
            # If samples are longer than the desired length, sliding and windowing them
            num_windows = current_length - desired_length + 1
            start_indices = np.arange(num_windows)
            window_indices = start_indices[:, None] + np.arange(desired_length)
            adjusted_samples = samples[:, window_indices, :].reshape(-1, desired_length, dim_features)
        elif find_strategy == 'last':
            adjusted_samples = samples[:, -desired_length:, :]
    return adjusted_samples


class PATH4TS(object):
    def __init__(self, config, zoo_path='', repr_path='', scaler_path='', device=None):
        """
            :param regression: whether regression
        """
        self.config = config
        self.device = device
        self.err_rate = config.err_rate
        self.components = math.ceil(config.test_pred_len / 12)
        self.repr_model_type = config.repr_model
        self.ensemble_size = config.ensemble_size

        self.model_zoo = sorted(self.config.model_zoo)

        with open(repr_path, 'rb') as f:
            print(repr_path)
            self.model_repr = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            self.scalers = pickle.load(f)

        self.model_reprs = np.array([self.model_repr[name] for name in self.model_zoo])


        # if self.repr_model_type == 'TS2Vec':
        #     self.repr_model = TS2Vec(input_dims=1, output_dims=128, device=self.device)
        #     self.repr_model.load(zoo_path)
        #     self.repr_model.net.to(self.device)
        # if self.repr_model_type == 'SimMTM':
        #     if config.data in ['m3', 'm4', 'tourism']:
        #         con = Config()
        #         con.seq_len = 24
        #         con.patch_len = 8
        #         con.stride = 2
        #         con.lm = 2
        #         # con.defaults['temperature'] = 0.05
        #         self.repr_model = SimMTM(con)
        #     else:
        #         self.repr_model = SimMTM(Config())

        #     self.repr_model.load(zoo_path)
        #     self.repr_model.to(self.device)
        #     self.repr_model.eval()
        if self.repr_model_type == 'SimMTM_zoo':
            if config.data in ['m3', 'm4', 'tourism']:
                con = Config()
                con.defaults['seq_len'] = 24
                self.repr_model = SimMTM_zoo(con)
            else:
                self.repr_model = SimMTM_zoo(Config())
            self.repr_model.load(zoo_path)
            self.repr_model.to(self.device)
            self.repr_model.eval()
        else:
            raise NotImplementedError

    def fit(self, samples, return_index=False, return_repr=False, use_norm=False):
        '''
            :param samples: [num_samples, timestamps, channels]
            :return: List[model_name]   e.g. ['c0', 'c1']
        '''

        N, T, C = samples.shape
        samples = samples.permute(0, 2, 1).reshape(N * C, T, 1)

        # if self.repr_model_type == 'TS2Vec':
        #     if use_norm:
        #         samples_norm = self.scalers.transform(samples.squeeze(axis=-1))
        #         samples_repr = self.repr_model.encode(samples_norm.unsqueeze(-1), encoding_window='full_series')# norm between-dims but not between-samples
        #     else:
        #         samples_repr = self.repr_model.encode(samples, encoding_window='full_series')  # n_samples x dims
        #     distances = cdist(samples_repr, self.model_reprs, metric='cosine')#


        if self.repr_model_type == 'SimMTM' or self.repr_model_type == 'SimMTM_zoo':
            if use_norm:
                samples_norm = self.scalers.transform(samples.squeeze(axis=-1))
                samples_repr = self.repr_model.extract_feature(samples_norm.unsqueeze(-1).float()) # norm between-dims but not between-samples
            else:
                samples_repr = self.repr_model.extract_feature(samples.float()).detach().cpu().numpy()  # n_samples x dims
            distances = cdist(samples_repr, self.model_reprs, metric='cosine')#
        else:
            raise NotImplementedError

        # Get the top-k nearest models' indices
        top_k_indices = np.argsort(distances, axis=1)[:, :self.ensemble_size]
        min_distances = np.min(distances, axis=1)
        if self.err_rate < 0:
            # Auto select based on interval
            l_interval = np.mean(min_distances) - 0.5 * np.std(min_distances)
            top_k_indices[min_distances < l_interval, :] = -1
        elif self.err_rate >= 0:
            top_k_indices[min_distances < self.err_rate, :] = -1
        
        # Reshape the indices to [k, num_samples, channels]
        top_k_indices = top_k_indices.T.reshape(self.ensemble_size, N, C)
        
        return top_k_indices
