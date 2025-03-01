import copy
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from exp.exp_resolver import resolve_experiment
from utils.configs import set_seed
from utils.arg_resolver import resolve_transformer_args, _model_is_transformer, setting_string, resolve_args, resolve_nbeats_args, resolve_dataset_args

import sys
sys.path.append("metalearned")

def parse():
    parser = argparse.ArgumentParser(
        description='Comparing performance of SeqFusion to other Time Series Benchmarks')

    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--itr', type=int, default=1, help='iteration times')
    parser.add_argument('--seed', type=int, default=2021, help='Random seed')

    # model settings
    parser.add_argument('--model', type=str, default='SeqFusion', help='model name')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=36, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=18, help='start token length')
    parser.add_argument('--pred_len', type=int, default=12, help='prediction sequence length')
    parser.add_argument('--time_budget', type=int, help='amount of time budget to train the model')
    parser.add_argument('--train_budget', type=int, help='length of training sequence')

    # data loader
    parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
    parser.add_argument('--source_data', type=str, default=None, help='dataset type')
    parser.add_argument('--target_data', type=str, default=None, help='dataset type')
    parser.add_argument('--root_path', type=str, default=None, help='root path of the data file')
    parser.add_argument('--data_path', type=str, default=None, help='data file')
    parser.add_argument('--target', type=str, default='OT', help='name of target column')
    parser.add_argument('--scale', type=bool, default=True, help='scale the time series with sklearn.StandardScale()')
    parser.add_argument('--features', type=str, default='S', help='name of target column')
    parser.add_argument('--enc_in', type=int, default=1)
    parser.add_argument('--dec_in', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seasonal_patterns', type=str, default='Hourly', help='subset for M4')
    parser.add_argument('--percent', type=int, default=10, help='subset for M4')
    parser.add_argument('--train_all', action='store_true', default=False, help='train all data')

    # ForecastPFN
    parser.add_argument('--model_path', type=str, default='s3://realityengines.datasets/forecasting/pretrained/gurnoor/models/20230202-025828/ckpts',
                        help='encoder input size')
    parser.add_argument('--scaler', type=str, default='standard',
                        help='scale the test series with sklearn.StandardScale()')

    # Metalearn
    parser.add_argument('--metalearn_freq', type=str,
                        help='which type of model should be used for the Metalearn model. Typically M, W, or D.')
    # patchTST
    parser.add_argument('--e_layers', type=int, default=None)
    parser.add_argument('--d_model', type=int, default=None)
    parser.add_argument('--patch_len', type=int, default=16)

    # SeqFusion
    parser.add_argument('--time_str', type=str, default=None, help='timestamp for log.')
    parser.add_argument('--load_model_path', type=str, default=None, help='path of the pre-trained model file, default is None for training from scratch.')
    parser.add_argument('--save_result_path', type=str, default='result_long_term_forecast.txt', help='path of the result file.')
    parser.add_argument('--mimo_block_nums', type=int, default=2, help='Number of SeqFusion blocks')
    parser.add_argument('--mimo_model_seq', type=int, nargs='+',default=[36, 12], help='seq_len/pred_len of SeqFusion blocks')
    parser.add_argument('--load_model_path_list', type=str, nargs='+',default=None, help='PTMs path of SeqFusion')
    parser.add_argument('--search_model', action='store_true', default=False, help='search components for SeqFusion')
    parser.add_argument('--select_one', action='store_true', default=False, help='search only one for SeqFusion')
    parser.add_argument('--use_norm', action='store_true', default=False, help='use global norm for SeqFusion')
    parser.add_argument('--seq_norm', action='store_true', default=False, help='use seq norm for zero-shot')
    parser.add_argument('--norm_horizon', action='store_true', default=False, help='use seq norm horizon for zero-shot')
    parser.add_argument('--basic_model', type=str, default='DLinear', help='Basic model of SeqFusion')
    parser.add_argument('--repr_model', type=str, default='TS2Vec', help='General extractor model of SeqFusion')
    parser.add_argument('--err_predictor', type=str, default='mean', help='Err predictor of SeqFusion.')
    parser.add_argument('--err_rate', type=float, default=0.0, help='Err threhold to use err predictor.')
    parser.add_argument('--ensemble_size', type=int, default=0, help='Number of ensemble SeqFusion blocks')


    parser.add_argument('--find_strategy', type=str, default='mean', help='mean or last')
    parser.add_argument('--cat_strategy', type=str, default='last', help='cat or last')
    parser.add_argument('--model_zoo', type=str, nargs='+',default=None, help='model zoo')
    parser.add_argument('--test_pred_len', type=int, default=None,
                        help='Testing prediction sequence length')

    parser.add_argument('--encoder_model_path', type=str, default='./SimMTM/SimMTM-ZOO_10000SimMTM-ZOO_128_mean_GT_LOSS_10000_True.pth')
    parser.add_argument('--model_zoo_repr_path', type=str, default='./SimMTM/Sampled_datasets_repr_10000SimMTM-ZOO_128_mean_GT_LOSS_10000_True.pkl')
    parser.add_argument('--scaler_path', type=str, default='./SimMTM/scaler_SimMTM-ZOO_128_mean_GT_LOSS_10000_True.pkl')
    return parser


def main():

    parser = parse()
    args = parser.parse_args()    
    set_seed(args.seed)

    args = resolve_args(args)
    args = resolve_dataset_args(args)
    if _model_is_transformer(args.model):
        args = resolve_transformer_args(args)
    if args.model == 'nbeats':
        args = resolve_nbeats_args(args)

    if args.model != 'ForecastPFN':
        args.model_name = None
    else:
        args.model_name = args.model_path.split('/')[-2]

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    if args.is_training:
        exp = resolve_experiment(args)
        for ii in range(args.itr):
            # setting record of experiments
            setting = setting_string(args, ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            torch.cuda.empty_cache()
            exp.reset()
    else:
        ii = 0
        setting = setting_string(args, ii)
        if args.model == 'SeqFusion' and args.search_model:
            from mptms.PATH4TS_M_ensemble import PATH4TS
            from utils.configs import MODEL_ZOO_PATH
            search_config = copy.deepcopy(args)

            search_exp = resolve_experiment(search_config)  # set experiments
            search_config.pred_len = args.test_pred_len
            search_config.batch_size = 1 if args.select_one else search_config.batch_size
            search_model = PATH4TS(args, zoo_path=args.encoder_model_path, repr_path=args.model_zoo_repr_path, scaler_path=args.scaler_path, device=search_exp.device)

            if search_config.data in ['m3', 'm4', 'tourism']:
                search_config.seq_padding_size = args.seq_len
                search_config.seq_len = args.test_seq_len
                test_data, test_loader = search_exp._get_data(flag='test', seq_padding_size=search_config.seq_padding_size)
            else:
                test_data, test_loader = search_exp._get_data(flag='test') 
            selected_model_list = []

            for i, (batch_x, _, _, _) in tqdm(enumerate(test_loader)):
                batch_x = batch_x.to(search_exp.device)
                search_model_list = search_model.fit(batch_x, return_index=True) # [num_samples, channel]
                if search_config.data in ['m3', 'm4', 'tourism'] and not args.select_one:
                    selected_model_list.append(search_model_list)
                else:
                    selected_model_list.extend(search_model_list)
                if args.select_one and (i >= args.train_budget - 1):
                    break

            args.load_model_path_list = [MODEL_ZOO_PATH[f'{args.basic_model}_' + i] for i in sorted(args.model_zoo)]

            if args.select_one:
                assert args.train_budget == 1
                from collections import Counter
                selected_model_list = np.array(selected_model_list)
                if args.ensemble_size >= 1:
                    selected_model_list = np.tile(selected_model_list, (1, len(test_data), 1))
                else:
                    channel_select = np.array([Counter(i).most_common(1)[0][0] for i in selected_model_list.transpose(1, 0)])
                    selected_model_list = np.tile(channel_select, (len(test_data), 1))
            elif args.data in ['m3', 'm4', 'tourism']:
                selected_model_list = np.concatenate(selected_model_list,axis=1)
            exp = resolve_experiment(args)
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            print('select shape:', selected_model_list.shape)
            exp.test(setting, test=1, selected_model_list=selected_model_list)
            torch.cuda.empty_cache()
        else:
            exp = resolve_experiment(args)
            exp.test(setting, test=1)
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
