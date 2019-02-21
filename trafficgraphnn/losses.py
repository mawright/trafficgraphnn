from keras import backend as K
from keras.losses import mean_absolute_error, mean_squared_error

def nan_masked_mse(y_true, y_pred):
    import tensorflow as tf
    is_nan = tf.is_nan(y_true)
    return _masked_mse(y_true, y_pred, is_nan)


def negative_masked_mse(y_true, y_pred):
    is_neg = K.less(y_true, 0.)
    return _masked_mse(y_true, y_pred, is_neg)


def negative_masked_mape(y_true, y_pred):
    is_neg = K.less(y_true, 0.)
    return _masked_mape(y_true, y_pred, is_neg)


def _masked_mse(y_true, y_pred, to_mask):
    mask_weight = 1 - K.cast(to_mask, K.floatx())
    sq_error = K.square(y_pred - y_true)
    return _apply_mask(sq_error, mask_weight)


def mean_square_error_veh(y_true, y_pred):
    return mean_squared_error(y_true[...,0], y_pred[...,0])


def negative_masked_mse_queue_length_m(y_true, y_pred):
    return negative_masked_mse(y_true[...,1], y_pred[...,1])


def scaled_two_feature_mse_constructor(scaling_weight=10):
    def scaled_two_feature_mse(y_true, y_pred):
        loss_1 = mean_square_error_veh(y_true, y_pred)
        loss_2 = negative_masked_mse_queue_length_m(y_true, y_pred)
        return scaling_weight * loss_1 + loss_2
    return scaled_two_feature_mse


def negative_masked_mae(y_true, y_pred):
    is_neg = K.less(y_true, 0.)
    return _masked_mae(y_true, y_pred, is_neg)


def mean_absolute_error_veh(y_true, y_pred):
    return mean_absolute_error(y_true[...,0], y_pred[...,0])


def negative_masked_mae_queue_length(y_true, y_pred):
    return negative_masked_mae(y_true[...,1], y_pred[...,1])


def _masked_mae(y_true, y_pred, to_mask):
    mask_weight = 1 - K.cast(to_mask, K.floatx())
    abs_error = K.abs(y_pred - y_true)
    return _apply_mask(abs_error, mask_weight)


def _masked_mape(y_true, y_pred, to_mask):
    mask_weight = 1 - K.cast(to_mask, K.floatx())
    num = K.abs(y_pred - y_true)
    den = K.clip(K.abs(y_true), K.epsilon(), None)
    percent_error = num / den
    return 100. * _apply_mask(percent_error, mask_weight)


def _apply_mask(error, mask_weights):
    masked_error = error * mask_weights
    sum_error = K.sum(masked_error)
    sum_weight = K.sum(mask_weights) + K.epsilon()
    return sum_error / sum_weight
