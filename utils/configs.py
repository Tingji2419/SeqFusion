import torch
import torch.backends.cudnn as cudnn
import os
import random
import numpy as np


def set_seed(seed):
    np.random.seed(seed=seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


DATA_CHANNEL = {
    'ETTh1': 7,
    'ETTh1-mean': 7,
    'ETTh2': 7,
    'ETTh2-mean': 7,
    'ETTm1': 7,
    'ETTm2': 7,
    'ECL': 321,
    'ECL-mean': 321,
    'traffic': 862,
    'traffic-mean': 862,
    'weather': 21,
    'weather-mean': 21,
    'exchange': 8,
    'illness': 7,
    'ili': 7,
    'Solar': 137,
    'PEMS03': 358,
    'PEMS04': 307,
    'PEMS07': 883,
    'PEMS08': 170,
    'm4': 1,
    'm3': 1,
    'm4_yearly': 1,
    'm4_quarterly': 1,
    'm4_monthly': 1,
    'm4_weekly': 1,
    'm4_daily': 1,
    'm4_hourly': 1,
    'm3_yearly': 1,
    'm3_quarterly': 1,
    'm3_monthly': 1,
    'm3_other': 1,
    'tourism_yearly': 1,
    'tourism_quarterly': 1,
    'tourism_monthly': 1,
    'web': 1, 
    'car': 1, 
    'hospital': 1,
}


DATA_PATH = {
    'exchange': ['./academic_data/exchange_rate/', 'exchange_rate.csv'],
    'ili': ['./academic_data/illness/', 'national_illness.csv'],
    'weather-mean': ['./academic_data/weather/', 'weather_agg.csv'],
    'traffic-mean': ['./academic_data/traffic/', 'traffic_agg.csv'],
    'ECL-mean': ['./academic_data/electricity/', 'electricity_agg.csv'],
    'ETTh1-mean': ['./academic_data/ETT-small/', 'ETTh1_agg.csv'],
    'ETTh2-mean': ['./academic_data/ETT-small/', 'ETTh2_agg.csv'],

    'weather': ['./academic_data/weather/', 'weather.csv'],
    'traffic': ['./academic_data/traffic/', 'traffic.csv'],
    'ECL': ['./academic_data/electricity/', 'electricity.csv'],
    'ETTh1': ['./academic_data/ETT-small/', 'ETTh1.csv'],
    'ETTh2': ['./academic_data/ETT-small/', 'ETTh2.csv'],
    'ETTm1': ['./academic_data/ETT-small/', 'ETTm1.csv'],
    'ETTm2': ['./academic_data/ETT-small/', 'ETTm2.csv'],
    'm4_yearly': ['./academic_data/m4/', 'm4_yearly_dataset.tsf'],
    'm4_quarterly': ['./academic_data/m4/', 'm4_quarterly_dataset.tsf'],
    'm4_monthly': ['./academic_data/m4/', 'm4_monthly_dataset.tsf'],
    'm4_weekly': ['./academic_data/m4/', 'm4_weekly_dataset.tsf'],
    'm4_daily': ['./academic_data/m4/', 'm4_daily_dataset.tsf'],
    'm4_hourly': ['./academic_data/m4/', 'm4_hourly_dataset.tsf'],
    'm3_yearly': ['./academic_data/m3/', 'm3_yearly_dataset.tsf'],
    'm3_quarterly': ['./academic_data/m3/', 'm3_quarterly_dataset.tsf'],
    'm3_monthly': ['./academic_data/m3/', 'm3_monthly_dataset.tsf'],
    'm3_other': ['./academic_data/m3/', 'm3_other_dataset.tsf'],
    'tourism_yearly': ['./academic_data/tourism/', 'tourism_yearly_dataset.tsf'],
    'tourism_quarterly': ['./academic_data/tourism/', 'tourism_quarterly_dataset.tsf'],
    'tourism_monthly': ['./academic_data/tourism/', 'tourism_monthly_dataset.tsf'],
    # **SAMPLED_DATA_PATH
}




MODEL_ZOO_PATH = {
'PatchTST_c55': './model_zoo_PatchTST_36_18_12_mse/PatchTST_c55.pth',
'PatchTST_c56': './model_zoo_PatchTST_36_18_12_mse/PatchTST_c56.pth',
'PatchTST_c57': './model_zoo_PatchTST_36_18_12_mse/PatchTST_c57.pth',
'PatchTST_c58': './model_zoo_PatchTST_36_18_12_mse/PatchTST_c58.pth',
'PatchTST_c59': './model_zoo_PatchTST_36_18_12_mse/PatchTST_c59.pth',
'PatchTST_c6': './model_zoo_PatchTST_36_18_12_mse/PatchTST_c6.pth',
'PatchTST_c60': './model_zoo_PatchTST_36_18_12_mse/PatchTST_c60.pth',
'PatchTST_c61': './model_zoo_PatchTST_36_18_12_mse/PatchTST_c61.pth',
'PatchTST_c62': './model_zoo_PatchTST_36_18_12_mse/PatchTST_c62.pth',
'PatchTST_c63': './model_zoo_PatchTST_36_18_12_mse/PatchTST_c63.pth',
'PatchTST_c64': './model_zoo_PatchTST_36_18_12_mse/PatchTST_c64.pth',
'PatchTST_c65': './model_zoo_PatchTST_36_18_12_mse/PatchTST_c65.pth',
'PatchTST_c66': './model_zoo_PatchTST_36_18_12_mse/PatchTST_c66.pth',
'PatchTST_c67': './model_zoo_PatchTST_36_18_12_mse/PatchTST_c67.pth',
'PatchTST_c68': './model_zoo_PatchTST_36_18_12_mse/PatchTST_c68.pth',
'PatchTST_c69': './model_zoo_PatchTST_36_18_12_mse/PatchTST_c69.pth',
'PatchTST_c7': './model_zoo_PatchTST_36_18_12_mse/PatchTST_c7.pth',
'PatchTST_c70': './model_zoo_PatchTST_36_18_12_mse/PatchTST_c70.pth',
'PatchTST_c71': './model_zoo_PatchTST_36_18_12_mse/PatchTST_c71.pth',
'PatchTST_c72': './model_zoo_PatchTST_36_18_12_mse/PatchTST_c72.pth',
'PatchTST_c73': './model_zoo_PatchTST_36_18_12_mse/PatchTST_c73.pth',
'PatchTST_c74': './model_zoo_PatchTST_36_18_12_mse/PatchTST_c74.pth',
'PatchTST_c75': './model_zoo_PatchTST_36_18_12_mse/PatchTST_c75.pth',
'PatchTST_c76': './model_zoo_PatchTST_36_18_12_mse/PatchTST_c76.pth',

'DLinear_d0': './model_zoo_DLinear_24_0_12_smape/DLinear_d0.pth',
'DLinear_d1': './model_zoo_DLinear_24_0_12_smape/DLinear_d1.pth',
'DLinear_d10': './model_zoo_DLinear_24_0_12_smape/DLinear_d10.pth',
'DLinear_d11': './model_zoo_DLinear_24_0_12_smape/DLinear_d11.pth',
'DLinear_d12': './model_zoo_DLinear_24_0_12_smape/DLinear_d12.pth',
'DLinear_d13': './model_zoo_DLinear_24_0_12_smape/DLinear_d13.pth',
'DLinear_d14': './model_zoo_DLinear_24_0_12_smape/DLinear_d14.pth',
'DLinear_d15': './model_zoo_DLinear_24_0_12_smape/DLinear_d15.pth',
'DLinear_d16': './model_zoo_DLinear_24_0_12_smape/DLinear_d16.pth',
'DLinear_d19': './model_zoo_DLinear_24_0_12_smape/DLinear_d19.pth',
'DLinear_d2': './model_zoo_DLinear_24_0_12_smape/DLinear_d2.pth',
'DLinear_d20': './model_zoo_DLinear_24_0_12_smape/DLinear_d20.pth',
'DLinear_d23': './model_zoo_DLinear_24_0_12_smape/DLinear_d23.pth',
'DLinear_d24': './model_zoo_DLinear_24_0_12_smape/DLinear_d24.pth',
'DLinear_d25': './model_zoo_DLinear_24_0_12_smape/DLinear_d25.pth',
'DLinear_d26': './model_zoo_DLinear_24_0_12_smape/DLinear_d26.pth',
'DLinear_d27': './model_zoo_DLinear_24_0_12_smape/DLinear_d27.pth',
'DLinear_d28': './model_zoo_DLinear_24_0_12_smape/DLinear_d28.pth',
'DLinear_d29': './model_zoo_DLinear_24_0_12_smape/DLinear_d29.pth',
'DLinear_d3': './model_zoo_DLinear_24_0_12_smape/DLinear_d3.pth',
'DLinear_d30': './model_zoo_DLinear_24_0_12_smape/DLinear_d30.pth',
'DLinear_d31': './model_zoo_DLinear_24_0_12_smape/DLinear_d31.pth',
'DLinear_d32': './model_zoo_DLinear_24_0_12_smape/DLinear_d32.pth',
'DLinear_d33': './model_zoo_DLinear_24_0_12_smape/DLinear_d33.pth',
'DLinear_d34': './model_zoo_DLinear_24_0_12_smape/DLinear_d34.pth',
'DLinear_d35': './model_zoo_DLinear_24_0_12_smape/DLinear_d35.pth',
'DLinear_d36': './model_zoo_DLinear_24_0_12_smape/DLinear_d36.pth',
'DLinear_d37': './model_zoo_DLinear_24_0_12_smape/DLinear_d37.pth',
'DLinear_d38': './model_zoo_DLinear_24_0_12_smape/DLinear_d38.pth',
'DLinear_d39': './model_zoo_DLinear_24_0_12_smape/DLinear_d39.pth',
'DLinear_d4': './model_zoo_DLinear_24_0_12_smape/DLinear_d4.pth',
'DLinear_d40': './model_zoo_DLinear_24_0_12_smape/DLinear_d40.pth',
'DLinear_d41': './model_zoo_DLinear_24_0_12_smape/DLinear_d41.pth',
'DLinear_d42': './model_zoo_DLinear_24_0_12_smape/DLinear_d42.pth',
'DLinear_d46': './model_zoo_DLinear_24_0_12_smape/DLinear_d46.pth',
'DLinear_d47': './model_zoo_DLinear_24_0_12_smape/DLinear_d47.pth',
'DLinear_d48': './model_zoo_DLinear_24_0_12_smape/DLinear_d48.pth',
'DLinear_d5': './model_zoo_DLinear_24_0_12_smape/DLinear_d5.pth',
'DLinear_d52': './model_zoo_DLinear_24_0_12_smape/DLinear_d52.pth',
'DLinear_d53': './model_zoo_DLinear_24_0_12_smape/DLinear_d53.pth',
'DLinear_d54': './model_zoo_DLinear_24_0_12_smape/DLinear_d54.pth',
'DLinear_d6': './model_zoo_DLinear_24_0_12_smape/DLinear_d6.pth',
'DLinear_d7': './model_zoo_DLinear_24_0_12_smape/DLinear_d7.pth',
'DLinear_d8': './model_zoo_DLinear_24_0_12_smape/DLinear_d8.pth',
'DLinear_d9': './model_zoo_DLinear_24_0_12_smape/DLinear_d9.pth',
}
