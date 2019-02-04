"""Layer utilities"""
import keras.backend as K

def batch_matmul(X, Y):
    """Keras-only implementation of tensorflow's batch matmul behavior in tf.matmul()"""
    with K.name_scope('batch_matmul'):
        x_shape = K.shape(X)
        y_shape = K.shape(Y)

        x_leading_dims = x_shape[:-2]
        y_leading_dims = y_shape[:-2]

        x_mat_dims = x_shape[-2:]
        y_mat_dims = y_shape[-2:]

        x_flatten_dim = K.prod(x_leading_dims, keepdims=True)
        y_flatten_dim = K.prod(y_leading_dims, keepdims=True)

        x_reshape_dims = K.concatenate([x_flatten_dim, x_mat_dims])
        y_reshape_dims = K.concatenate([y_flatten_dim, y_mat_dims])

        product_dims = K.concatenate([x_leading_dims,
                                    x_mat_dims[:1],
                                    y_mat_dims[1:]])

        X = K.reshape(X, x_reshape_dims)
        Y = K.reshape(Y, y_reshape_dims)

        prod = K.batch_dot(X, Y)

        return K.reshape(prod, product_dims)
