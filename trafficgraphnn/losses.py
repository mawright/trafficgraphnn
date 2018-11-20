from keras import backend as K
from keras.losses import mean_absolute_error

def nan_masked_mse(y_true, y_pred):
    import tensorflow as tf
    is_nan = tf.is_nan(y_true)
    return _masked_mse(y_true, y_pred, is_nan)


def negative_masked_mse(y_true, y_pred):
    is_neg = K.less(y_true, 0.)
    return _masked_mse(y_true, y_pred, is_neg)


def _masked_mse(y_true, y_pred, to_mask):
    mask_weight = 1 - K.cast(to_mask, K.floatx())
    sq_error = K.square(y_pred - y_true)
    masked_error = sq_error * mask_weight
    return K.sum(masked_error) / K.sum(mask_weight)


def negative_masked_mae(y_true, y_pred):
    is_neg = K.less(y_true, 0.)
    return _masked_mae(y_true, y_pred, is_neg)


def mean_absolute_error_veh(y_true, y_pred):
    return mean_absolute_error(y_true[...,0], y_pred[...,0])


def negative_masked_mae_queue_length_m(y_true, y_pred):
    return negative_masked_mae(y_true[...,1], y_pred[...,1])


def _masked_mae(y_true, y_pred, to_mask):
    mask_weight = 1 - K.cast(to_mask, K.floatx())
    abs_error = K.abs(y_pred - y_true)
    masked_error = abs_error * mask_weight
    # sum_error = K.sum(masked_error, axis=-1)
    # sum_weight = K.sum(mask_weight, axis=-1)
    # return K.switch(sum_weight == 0., 0, sum_error / sum_weight)
    return K.sum(masked_error) / K.sum(mask_weight)
