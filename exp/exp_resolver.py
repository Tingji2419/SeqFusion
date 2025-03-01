import warnings
from exp.exp_transformer import Exp_Transformer
# from exp.exp_transformer_metalearn import Exp_Transformer_Meta
from exp.exp_ForecastPFN import Exp_ForecastPFN
from exp.exp_arima import Exp_Arima
from exp.exp_prophet import Exp_Prophet
from exp.exp_metalearn import Exp_Metalearn
from exp.exp_mean import Exp_Mean
from exp.exp_last import Exp_Last
from exp.exp_seasonalNaive import Exp_SeasonalNaive
from exp.exp_SeqFusion import Exp_SeqFusion
from exp.exp_nbeats import Exp_Nbeats
from exp.exp_transfer import Exp_Transfer


warnings.filterwarnings('ignore')


def resolve_experiment(args):
    if args.target_data is not None:
        return Exp_Transfer(args)
    exp_dict = {
        'FEDformer': Exp_Transformer,
        'FEDformer-w': Exp_Transformer,
        'FEDformer-f': Exp_Transformer,
        'Autoformer': Exp_Transformer,
        'Transformer': Exp_Transformer,
        'Informer': Exp_Transformer,
        'ForecastPFN': Exp_ForecastPFN,
        'Arima': Exp_Arima,
        'Prophet': Exp_Prophet,
        'Metalearn': Exp_Metalearn,
        'Mean': Exp_Mean,
        'Last': Exp_Last,
        'SeasonalNaive': Exp_SeasonalNaive,
        'DLinear': Exp_Transformer,
        'SeqFusion': Exp_SeqFusion,
        'nbeats': Exp_Nbeats,
    }
    return exp_dict[args.model](args)
