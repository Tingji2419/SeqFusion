import numpy as np
import tensorflow as tf
from keras import backend


def RSE(pred, true):
    return np.sqrt(np.nansum((true - pred) ** 2)) / np.sqrt(np.nansum((true - true.nanmean()) ** 2))


def CORR(pred, true):
    u = ((true - true.nanmean(0)) * (pred - pred.nanmean(0))).nansum(0)
    d = np.sqrt(((true - true.nanmean(0)) ** 2 * (pred - pred.nanmean(0)) ** 2).nansum(0))
    return (u / d).nanmean(-1)


def MAE(pred, true):
    return np.nanmean(np.abs(pred - true))


def MSE(pred, true):
    return np.nanmean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.nanmean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.nanmean(np.square((pred - true) / true))

def SMAPE(pred, true):
    return np.nanmean(200 * np.abs(pred - true) / (np.abs(pred) + np.abs(true) + 1e-8))

def ND(pred, true):
    return np.nanmean(np.abs(true - pred)) / np.nanmean(np.abs(true))


def metric_with_nan(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    smape = SMAPE(pred, true)
    nd = ND(pred, true)

    return mae, mse, rmse, mape, smape, mspe, nd

def smape(y_true, y_pred):
    """ Calculate Armstrong's original definition of sMAPE between `y_true` & `y_pred`.
        `loss = 200 * mean(abs((y_true - y_pred) / (y_true + y_pred), axis=-1)`
        Args:
        y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
        Returns:
        Symmetric mean absolute percentage error values. shape = `[batch_size, d0, ..
        dN-1]`.
        """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    diff = tf.abs(
        (y_true - y_pred) /
        backend.maximum(y_true + y_pred, backend.epsilon())
    )
    return 200.0 * backend.nanmean(diff, axis=-1)
