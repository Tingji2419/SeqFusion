from data_provider.data_loader import Dataset_Custom, Dataset_TSF
from torch.utils.data import DataLoader
# from metalearned.resources.electricity.dataset import ElectricityDataset, ElectricityMeta
# from metalearned.resources.m3.dataset import M3Dataset, M3Meta
# from metalearned.resources.m4.dataset import M4Dataset, M4Meta
# from metalearned.resources.tourism.dataset import TourismDataset, TourismMeta
# from metalearned.resources.traffic.dataset import TrafficDataset, TrafficMeta

data_dict = {
    'custom': Dataset_Custom,
    'ili': Dataset_Custom,
    'exchange': Dataset_Custom,
    'ECL': Dataset_Custom,
    'weather': Dataset_Custom,
    'traffic': Dataset_Custom,
    'ETTh1': Dataset_Custom,
    'ETTh2': Dataset_Custom,
    'ETTm1': Dataset_Custom,
    'ETTm2': Dataset_Custom,
    'ECL-mean': Dataset_Custom,
    'weather-mean': Dataset_Custom,
    'traffic-mean': Dataset_Custom,
    'ETTh1-mean': Dataset_Custom,
    'ETTh2-mean': Dataset_Custom,
    'ETTm1-mean': Dataset_Custom,
    'ETTm2-mean': Dataset_Custom,
    'beer': Dataset_Custom,
    'climate': Dataset_Custom,
    'stock': Dataset_Custom,
    'temp': Dataset_Custom,
    'population': Dataset_Custom,
    'gold': Dataset_Custom,
    'website': Dataset_Custom,
    'bitcoin': Dataset_Custom,
    # 'm3': M3Dataset,
    'm3': Dataset_TSF,
    'm4': Dataset_TSF,
    # 'electricity': ElectricityDataset,
    'tourism': Dataset_TSF,
    # 'traffic': TrafficDataset,
}


def data_provider(args, flag, seq_padding_size=None):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        scaler=args.scaler,
        scale=args.scale,
        train_budget=args.train_budget,
        seasonal_patterns=args.seasonal_patterns,
        data=args.data,
        percent=args.percent,
        train_all=args.train_all,
        seq_padding_size=seq_padding_size
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
