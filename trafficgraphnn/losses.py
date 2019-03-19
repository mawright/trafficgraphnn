from keras import backend as K


def huber(y_true, y_pred, delta=1):
    abs_error = K.abs(y_true - y_pred)
    loss = _huber_helper(abs_error, delta)
    return K.sum(loss, axis=-1)


def _masked_huber(y_true, y_pred, to_mask, delta=1):
    mask_weight = 1 - K.cast(to_mask, K.floatx())
    abs_error = K.abs(y_true - y_pred)
    loss = _huber_helper(abs_error, delta)
    return _apply_mask(loss, mask_weight)


def negative_masked_huber(y_true, y_pred, delta=1):
    is_neg = K.less(y_true, 0.)
    return _masked_huber(y_true, y_pred, is_neg, delta=delta)


def _huber_helper(abs_error, delta):
    quadratic = K.minimum(abs_error, delta)
    linear = abs_error - quadratic
    loss = .5 * K.square(quadratic) + linear * delta
    return loss


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


def negative_masked_mae(y_true, y_pred):
    is_neg = K.less(y_true, 0.)
    return _masked_mae(y_true, y_pred, is_neg)


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
