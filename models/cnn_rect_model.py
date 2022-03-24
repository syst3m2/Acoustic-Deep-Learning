"""
    Andrew Pfau
    Sonar Classifier
    
    This file holds the baseline custom cnn model used for testing different model architectures of CNNs
    Split into a file seperate from model.py because it changes more frequently

    This is a CNN with rectangular kernels and 5 blocks

"""
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dense
from tensorflow.keras.layers import AvgPool2D, GlobalAveragePooling2D, MaxPool2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import ReLU, concatenate
import tensorflow.keras.backend as K


def dev_model(input_shape, num_classes, activation_fcn, hparam=None):
    """   
        A model to test different models, number of filters, layer structure, and kernels
        Modified to use tensorboard's hparams plugin
        Hparams is a list of hyperparameters to test
    """
    
    filters = 32
    
    
    #batch norm + relu + conv
    def bn_rl_conv(x,filters,kernel=1,strides=1):
        
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(filters, kernel, strides=strides,padding = 'same')(x)
        return x
    
    def dense_block(x, repetition):
        
        for _ in range(repetition):
            y = bn_rl_conv(x, 4*filters)
            y = bn_rl_conv(y, filters, (10,5))
            x = concatenate([y,x])
        return x
        
    def transition_layer(x):
        
        x = bn_rl_conv(x, K.int_shape(x)[-1] //2 )
        x = AvgPool2D(2, strides = 2, padding = 'same')(x)
        return x
    
    input = Input (input_shape)
    x = Conv2D(64, 7, strides = 2, padding = 'same')(input)
    x = MaxPool2D(3, strides = 2, padding = 'same')(x)
    
    for repetition in [6,12,24,16]:
        
        d = dense_block(x, repetition)
        x = transition_layer(d)
    x = GlobalAveragePooling2D()(d)
    output = Dense(num_classes, activation = 'softmax')(x)
    
    model = Model(input, output)
    return model





def build_model(param, input_shape, activation_fcn, hparams=None):
    # can alter this later to handle a hparam search case and non hpararm case 
    return dev_model(input_shape, param.num_classes, activation_fcn, hparams)
