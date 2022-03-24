# file to convert command line parameters into model optimizer function

import tensorflow as tf

def get_optimizer(params):
    gpus = len(tf.config.list_physical_devices('GPU')) 

    if params.optimizer == 'adam':
        if gpus > 1:
            return tf.keras.optimizers.Adam(learning_rate=params.learning_rate_start)
        else:
            return tf.keras.optimizers.Adam(learning_rate=params.learning_rate_start,clipnorm=1)
    elif params.optimizer == 'sgd':
        if gpus > 1:
            return tf.keras.optimizers.SGD(learning_rate=params.learning_rate_start, momentum=0.9)
        else:
            return tf.keras.optimizers.SGD(learning_rate=params.learning_rate_start, momentum=0.9,clipnorm=1)
    elif params.optimizer == 'adagrad':
        return tf.keras.optimizers.Adagrad(learning_rate=params.learning_rate_start,)
def get_loss(loss, label_smooth):
    if loss == 'binarycrossentropy':
        return tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smooth)
    elif loss == 'crossentropy':
        return tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smooth)
    elif loss == 'mse':
        return tf.keras.losses.MeanSquaredError()