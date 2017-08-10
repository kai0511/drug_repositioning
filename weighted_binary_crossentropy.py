import keras.backend as K
import numpy as np

def weighted_binary_crossentropy(y_pred, y_true, weight_dict):
    ''' Args: y_pred and y_true are n*1 or n*2 tensor flow array.
    weights_dict python dictionary mapping class labels to their corresponding weights.
    '''
    if K.ndim(y_true) == 1:
            y_true = K.reshape(y_true, (K.shape(y_true)[0], 1))
            y_pred = K.reshape(y_pred, (K.shape(y_pred)[0], 1))

        if K.int_shape(y_true)[1] == 1:
            labels = K.concatenate([K.ones(K.shape(y_true)) - y_true, y_true], axis = 1)
            y_pred = K.concatenate([K.ones(K.shape(y_pred)) - y_pred, y_pred], axis = 1)

        array_labels = K.get_value(labels)
        sample_weights = np.ones(array_labels.shape)
        sample_weights[array_labels == 1] = weight_dict[1]
        sample_weights[array_labels == 0] = weight_dict[0]

        loss = -K.sum(labels * K.log(y_pred), axis = 1)
        return K.dot(loss, K.variable(sample_weights))
