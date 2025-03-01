from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils.configs import DATA_PATH, DATA_CHANNEL
def _model_is_transformer(model):
    if model in ['FEDformer', 'FEDformer-f', 'FEDformer-w', 'FEDformer_Meta', 'Autoformer', 'Informer', 'Transformer', 'DLinear','SeqFusion', 'PatchTST']:
        return True
    return False

def setting_string(args, ii):
    setting = '{}_{}_sl{}_ll{}_pl{}_timebudget_{}_trainbudget_{}_model-path_{}_{}_feature_{}_seasonal_{}_itr_{}'.format(
        args.model,
        args.data,
        args.seq_len,
        args.label_len,
        args.test_pred_len if args.test_pred_len else args.pred_len,
        args.time_budget,
        args.train_budget,
        args.model_name,
        args.time_str,
        args.features,
        args.seasonal_patterns,
        ii)
    return setting


def resolve_args(args):

    args.freq = 'h'
    args.checkpoints = './checkpoints/'
    args.embed = 'timeF'
    # args.batch_size = 32
    # args.use_multi_gpu = False
    # args.devices = '0,1'
    args.num_workers = 10
    if not args.root_path and not args.data_path:
        if args.data in ['m4', 'm3', 'tourism']:
            data_full_name = args.data + '_' + args.seasonal_patterns.lower()
            args.root_path, args.data_path = DATA_PATH[data_full_name][0], DATA_PATH[data_full_name][1]
        else:
            args.root_path, args.data_path = DATA_PATH[args.data][0], DATA_PATH[args.data][1]
        print('Auto load data: ', args.root_path, args.data_path)
    if args.scaler == 'standard':
        args.scaler = StandardScaler()
    if args.scaler == 'minmax':
        args.scaler = MinMaxScaler()
    return args

def resolve_dataset_args(args):
    if args.data in ['m4', 'm3', 'tourism']:
        args.loss = 'smape'
        args.scale = False
    else:
        args.loss = 'mse'
    
    if args.model == 'SeqFusion' and args.data in ['m4', 'm3', 'tourism']:
        # for SeqFusion searching
        from data_provider.m4 import M4Meta, M3Meta, TourismMeta
        if args.data == 'm4':
            args.test_pred_len = M4Meta.horizons_map[args.seasonal_patterns]
            args.test_seq_len = M4Meta.lookback_window_map[args.seasonal_patterns]

        if args.data == 'm3':
            args.test_pred_len = M3Meta.horizons_map[args.seasonal_patterns]
            args.test_seq_len = M3Meta.lookback_window_map[args.seasonal_patterns]

        if args.data == 'tourism':
            args.test_pred_len = TourismMeta.horizons_map[args.seasonal_patterns]
            args.test_seq_len = TourismMeta.lookback_window_map[args.seasonal_patterns]

    if args.target_data is None:
        return args

    from data_provider.m4 import M4Meta, M3Meta, TourismMeta
    if args.target_data == 'm4':
        # args.pred_len = M4Meta.horizons_map[args.seasonal_patterns]
        # args.seq_len = M4Meta.lookback_window_map[args.seasonal_patterns]
        # args.label_len = 0
        if args.target_data != args.data:
            args.source_freq = M4Meta.seasonal_map[args.seasonal_patterns]
    if args.target_data == 'm3':
        # args.pred_len = M3Meta.horizons_map[args.seasonal_patterns]
        # args.seq_len = M3Meta.lookback_window_map[args.seasonal_patterns]
        # args.label_len = 0
        if args.target_data != args.data:
            args.source_freq = M3Meta.seasonal_map[args.seasonal_patterns]
    if args.target_data == 'tourism':
        # args.pred_len = TourismMeta.horizons_map[args.seasonal_patterns]
        # args.seq_len = TourismMeta.lookback_window_map[args.seasonal_patterns]
        # args.label_len = 0
        if args.target_data != args.data:
            args.source_freq = TourismMeta.seasonal_map[args.seasonal_patterns]

    return args

def resolve_transformer_args(args):
    args.mode_select = 'random'
    args.modes = 64
    args.L = 3
    args.base = 'legendre'
    args.cross_activation = 'tanh'

    # args.enc_in = 1
    if args.features == 'M' and args.enc_in == 1:
        args.dec_in = args.enc_in = DATA_CHANNEL[args.data]
        print('Data Channel: ', args.enc_in,DATA_CHANNEL[args.data],args.data)
    else:
        args.dec_in = args.enc_in
    if args.features == 'M':
        args.c_out = DATA_CHANNEL[args.data]
    else:
        args.c_out = 1
    args.d_model = 512 if not args.d_model else args.d_model
    args.n_heads = 8
    args.e_layers = 2 if not args.e_layers else args.e_layers
    args.d_layers = 1
    args.d_ff = 2048
    args.moving_avg = [25]
    args.factor = 3
    args.distil = True
    args.dropout = 0.05
    args.activation = 'gelu'
    args.output_attention = False
    args.do_predict = False
    args.train_epochs = 10
    args.patience = 3
    args.learning_rate = 0.0001
    args.des = 'Exp'
    args.lradj = 'type1'
    args.use_amp = False

    if args.model == 'FEDformer-w':
        args.version = 'Wavelet'
    elif args.model == 'FEDformer-f':
        args.version = 'Fourrier'
    elif 'FEDformer' in args.model:
        args.version = 'Wavelet'

    return args

def resolve_nbeats_args(args):
    if args.features == 'M' and args.enc_in != 1:
        args.dec_in = args.enc_in = DATA_CHANNEL[args.data]
    else:
        args.dec_in = args.enc_in
    if args.features == 'M':
        args.c_out = DATA_CHANNEL[args.data]
    else:
        args.c_out = 1
    args.e_layers = 30
    args.d_layers = 4
    args.d_model = 512
    args.patience = 3
    args.learning_rate = 0.0001
    args.des = 'Exp'
    args.lradj = 'type1'
    args.use_amp = False
    args.train_epochs = 20
    args.output_attention = False
    return args
